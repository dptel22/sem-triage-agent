"""Gate 0 Part B — Meta Model API verification harness.

Real empirical tests against https://api.meta.ai/v1 (OpenAI-compatible):
  --mode smoke         one image: /models probe, SAM text prompt, SAM box-prompt
                       probe (official wire format), cropped-ROI text prompt,
                       muse-spark-1.3 tool-calling check. Prints raw responses.
  --mode benchmark20   20-image Carinthia-S benchmark across SAM arms with
                       IoU/Dice vs ground-truth masks; writes
                       results/gate0/sam_20_results.json.

No mocks, no fabricated numbers: every network call goes to the real endpoint
and every error is reported verbatim. Mask decoding uses src/api/sam_decoder.py
(a wrapper around the official meta-sam-parser reference implementation).

Env: META_API_KEY (or MODEL_API_KEY). Requires: openai, numpy, pillow,
meta-sam-parser, opencv-python-headless (CLAHE arm only).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import sam_decoder  # noqa: E402

import openai  # noqa: E402

API_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
MUSE_MODEL = "muse-spark-1.3"
OUT_DIR = PROJECT_ROOT / "results" / "gate0"

NOMINAL_CLASSES = {3, 4, 6}
RARE_CLASSES = {1, 2, 5}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def resolve_api_key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        print(
            "FATAL: no API key found. Set META_API_KEY (or MODEL_API_KEY) in the "
            "environment before running. Refusing to mock or fabricate results.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return key


def validate_base_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"FATAL: base URL must be https, got {url!r}")
    host = (parsed.hostname or "").lower()
    if not host or host in ("localhost",) or host.endswith(".local") or host.endswith(
        ".internal"
    ):
        raise SystemExit(f"FATAL: refusing private/localhost host {host!r}")
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        octets = [int(p) for p in parts]
        if octets[0] in (10, 127) or (
            octets[0] == 192 and octets[1] == 168
        ) or (octets[0] == 172 and 16 <= octets[1] <= 31):
            raise SystemExit(f"FATAL: refusing private IP host {host!r}")


def make_client(api_key: str) -> openai.OpenAI:
    validate_base_url(API_BASE_URL)
    return openai.OpenAI(api_key=api_key, base_url=API_BASE_URL, timeout=180.0)


def image_to_data_url(path: Path) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as im:
        im.load()
        fmt = (im.format or "PNG").lower().replace("jpeg", "jpeg")
        width, height = im.size
        buf_path = path
        if fmt not in ("png", "jpeg", "webp", "gif"):
            fmt = "png"
            tmp = path.with_suffix(".tmp_gate0.png")
            im.convert("RGB").save(tmp, format="PNG")
            buf_path = tmp
        data = base64.b64encode(buf_path.read_bytes()).decode("ascii")
        mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}[fmt]
    return f"data:{mime};base64,{data}", (width, height)


def print_verbatim(label: str, text: str | None, cap: int = 2000) -> None:
    if text is None:
        print(f"{label}: <None>")
        return
    print(f"{label} ({len(text)} chars):")
    if len(text) <= cap:
        print(text)
    else:
        head, tail = cap // 2, cap // 2
        print(text[:head])
        print(f"  [... VERBATIM-TRUNCATED {len(text) - cap} CHARS; full text saved to JSON ...]")
        print(text[-tail:])


def err_text(error: BaseException) -> str:
    parts = [f"{type(error).__name__}: {error}"]
    for attr in ("status_code", "code", "body", "request_id"):
        value = getattr(error, attr, None)
        if value is not None and attr not in ("body",) or (attr == "body" and value):
            parts.append(f"{attr}={value!r}")
    return " | ".join(parts)


RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


def with_retries(fn, *, what: str, retries: int):
    for attempt in range(retries + 1):
        try:
            return fn()
        except RETRYABLE as error:
            if attempt == retries:
                raise
            wait = 1.5 * (2**attempt)
            print(f"  [retry {attempt + 1}/{retries}] {type(error).__name__} on {what}; "
                  f"waiting {wait:.0f}s: {error}")
            time.sleep(wait)


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0, 1.0  # both empty
    iou = inter / union
    denom = int(pred.sum()) + int(gt.sum())
    dice = (2 * inter) / denom if denom else 1.0
    return iou, dice


def masks_to_canvas(parsed: sam_decoder.SamParsedResponse, height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for mask in parsed.masks:
        canvas = np.logical_or(canvas, sam_decoder.mask_to_full_canvas(mask, height, width))
    return canvas.astype(np.uint8)


def load_gt_mask(mask_path: Path) -> np.ndarray:
    """Binarize a Carinthia-S mask with the manifest's foreground criterion.

    These masks are soft-edged (anti-aliased 0..255); Agent 1's manifest defines
    foreground as ``> 0``, which reproduces ``foreground_pixel_count`` on 20/20
    samples and makes ``tight_bbox_xyxy``/``padded_bbox_xyxy`` exact inclusive
    boxes (verified empirically before the benchmark).
    """
    return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)


def crop_box(item: dict, width: int, height: int, args) -> tuple[int, int, int, int, str]:
    """Half-open crop box for Arm 3 plus its provenance label.

    Prefers Agent 1's precomputed ``padded_bbox_xyxy``; otherwise pads the tight
    box by max(16px, 25%) per side and clamps to the image.
    """
    box = item.get("padded_box") or item["box"]
    source = "padded_bbox_xyxy" if item.get("padded_box") else "computed_25pct"
    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if args.box_convention == "inclusive":
        bx2, by2 = bx2 + 1, by2 + 1  # inclusive wire coords -> exclusive crop bounds
    if source == "computed_25pct":
        pad_x = max(16, int(0.25 * (bx2 - bx1)))
        pad_y = max(16, int(0.25 * (by2 - by1)))
        bx1, by1, bx2, by2 = bx1 - pad_x, by1 - pad_y, bx2 + pad_x, by2 + pad_y
    cx1, cy1 = max(0, bx1), max(0, by1)
    cx2, cy2 = min(width, bx2), min(height, by2)
    return cx1, cy1, cx2, cy2, source


def sam_call(client: openai.OpenAI, *, image_data_url: str, prompt_text: str,
             include_confidence: bool = False, retries: int = 2):
    """One unary SAM 3.1 Responses call. Returns the raw Response object."""
    metadata = {"include_confidence": "true"} if include_confidence else None
    kwargs = dict(
        model=SAM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )
    if metadata:
        kwargs["metadata"] = metadata
    return with_retries(lambda: client.responses.create(**kwargs), what=f"sam:{prompt_text[:40]!r}",
                        retries=retries)


def decode_sam_response(resp) -> tuple[sam_decoder.SamParsedResponse, str | None]:
    text = getattr(resp, "output_text", None)
    if text is None or text == "":
        return sam_decoder.SamParsedResponse(raw_output=text or ""), None
    try:
        return sam_decoder.decode_output_text(text), None
    except Exception as error:  # decode failure is a finding, not something to hide
        return sam_decoder.SamParsedResponse(raw_output=text), err_text(error)


# ---------------------------------------------------------------------------
# Manifest loading (Agent 1's file) + synthetic smoke fixture
# ---------------------------------------------------------------------------

IMAGE_KEYS = ("image", "image_path", "rgb_png_path", "file", "filename", "png")
MASK_KEYS = ("mask", "mask_path", "original_mask_path", "gt_mask", "gt_mask_path", "label_mask")
CLASS_KEYS = ("class_id", "class_label", "class", "label", "defect_class", "category")
BOX_KEYS = ("box", "tight_bbox_xyxy", "padded_bbox_xyxy", "bbox", "bbox_xyxy", "box_xyxy", "gt_box")


def _first_key(entry: dict, keys) -> object:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else raw.get("samples") or raw.get("images") or raw.get("entries") or raw.get("data")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(
            f"FATAL: could not find an image list in {path}. Top-level type: "
            f"{type(raw).__name__}; keys: {list(raw) if isinstance(raw, dict) else 'n/a'}. "
            "Inspect the file and adjust load_manifest()."
        )
    resolved = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"FATAL: manifest entry #{index} is {type(entry).__name__}, not a dict.")
        image = _first_key(entry, IMAGE_KEYS)
        mask = _first_key(entry, MASK_KEYS)
        if image is None or mask is None:
            raise SystemExit(
                f"FATAL: manifest entry #{index} lacks image/mask keys. Observed keys: "
                f"{sorted(entry)}. Adjust IMAGE_KEYS/MASK_KEYS in this script."
            )
        def _resolve_file(rel: object) -> Path:
            p = Path(str(rel))
            if p.is_file():
                return p.resolve()
            if (PROJECT_ROOT / p).is_file():
                return (PROJECT_ROOT / p).resolve()
            if (path.parent / p).is_file():
                return (path.parent / p).resolve()
            return (path.parent / p).resolve()

        image_path = _resolve_file(image)
        mask_path = _resolve_file(mask)
        for p in (image_path, mask_path):
            if not p.is_file():
                raise SystemExit(f"FATAL: manifest entry #{index} references missing file: {p}")
        class_id = _first_key(entry, CLASS_KEYS)
        box = _first_key(entry, BOX_KEYS)
        resolved.append(
            {
                "id": entry.get("id", entry.get("name", f"entry_{index:02d}")),
                "image_path": image_path,
                "mask_path": mask_path,
                "class_id": int(class_id) if class_id is not None else None,
                "box": [float(v) for v in box] if box else None,
                # Agent 1's manifest also carries a precomputed padded crop box
                # (used by Arm 3); fall back to computed 25% padding when absent.
                "padded_box": [float(v) for v in entry["padded_bbox_xyxy"]]
                if "padded_bbox_xyxy" in entry else None,
            }
        )
    return resolved


def make_smoke_fixture(out_dir: Path) -> dict:
    """Deterministic synthetic input used ONLY while Agent 1's manifest is absent.

    The fixture is a test INPUT (clearly labeled); every API response is still
    real. A dark scratch (ground truth) plus a bright speckle distractor.
    """
    rng = np.random.default_rng(42)
    height, width = 480, 640
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    img += rng.integers(-12, 12, size=img.shape).astype(np.uint8)  # brushed-metal noise
    mask = np.zeros((height, width), dtype=np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    # Scratch: dark diagonal stroke from (150, 100) to (330, 520), ~6px wide.
    x0, y0, x1, y1 = 100.0, 150.0, 520.0, 330.0
    t = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0)) /
                ((x1 - x0) ** 2 + (y1 - y0) ** 2), 0, 1)
    px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    dist = np.hypot(xx - px, yy - py)
    scratch = dist <= 3.0
    img[scratch] = (40, 38, 36)
    mask[scratch] = 1

    # Distractor: bright specular blob (NOT in ground truth).
    blob = (xx - 480) ** 2 + (yy - 120) ** 2 <= 12**2
    img[blob] = (240, 240, 235)

    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "smoke_001.png"
    mask_path = out_dir / "smoke_001_mask.png"
    Image.fromarray(img).save(image_path)
    Image.fromarray(mask * 255).save(mask_path)
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # inclusive, like the manifest
    return {"id": "smoke_001_synthetic", "image_path": image_path, "mask_path": mask_path,
            "class_id": 2, "box": box}


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------

MUSE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_defect_triage",
        "description": "Record the visual triage verdict for one industrial SEM part image.",
        "parameters": {
            "type": "object",
            "properties": {
                "morphology_type": {"type": "string",
                                    "description": "Dominant defect morphology, e.g. "
                                                   "'scratch', 'particle', 'contamination', 'none'."},
                "confidence": {"type": "number",
                               "description": "Self-reported confidence from 0.0 to 1.0."},
                "action": {"type": "string", "enum": ["pass", "review", "scrap"],
                           "description": "Recommended next action for the part."},
            },
            "required": ["morphology_type", "confidence", "action"],
        },
    },
}


def run_smoke(client: openai.OpenAI, item: dict, args) -> dict:
    results: dict = {"mode": "smoke", "fixture": str(item["image_path"]),
                     "is_synthetic_fixture": bool(args.make_fixture),
                     "image_size": None, "box_convention": args.box_convention}
    data_url, (width, height) = image_to_data_url(item["image_path"])
    has_gt = bool(item.get("has_gt", True))
    gt = load_gt_mask(item["mask_path"]) if has_gt else None
    box = item["box"]
    results["image_size"] = [width, height]
    results["ground_truth_box_half_open"] = box
    results["gt_bin_threshold"] = ">0" if has_gt else None
    results["gt_mask_pixels"] = int(gt.sum()) if gt is not None else None

    print("=" * 78)
    print(f"SMOKE on {item['image_path'].name}  size={width}x{height}  "
          f"gt_mask_pixels={results['gt_mask_pixels']}  box(half-open)={box}")
    print("=" * 78)

    # Step 0: models list (cheap ground truth about what this key can see).
    print("\n--- [0] GET /models ---")
    try:
        models = with_retries(lambda: list(client.models.list()), what="models.list", retries=1)
        ids = sorted(str(m.id) for m in models)
        results["models_listed"] = ids
        print("model ids:", ids)
        print("sam-3.1 listed:", SAM_MODEL in ids, "| muse-spark-1.3 listed:", MUSE_MODEL in ids)
    except Exception as error:
        results["models_listed"] = None
        results["models_listed_error"] = err_text(error)
        print("models.list FAILED:", err_text(error))

    # Step 1: SAM text prompt, full image.
    print(f"\n--- [1] SAM {SAM_MODEL} TEXT PROMPT (full image, text='defect') ---")
    results["arm1_full_text"] = {}
    try:
        resp = sam_call(client, image_data_url=data_url, prompt_text="defect",
                        include_confidence=args.include_confidence, retries=args.retries)
        text = getattr(resp, "output_text", None)
        print(f"resp.id={getattr(resp, 'id', None)} model={getattr(resp, 'model', None)} "
              f"status={getattr(resp, 'status', None)}")
        print_verbatim("RAW resp.output_text", text)
        parsed, decode_err = decode_sam_response(resp)
        results["arm1_full_text"] = {
            "response_id": getattr(resp, "id", None),
            "raw_output_text": text,
            "n_box_records": len(parsed.boxes),
            "n_mask_records": len(parsed.masks),
            "boxes_half_open": [ [b.left, b.top, b.right, b.bottom] for b in parsed.boxes ],
            "diagnostics": [f"{d.severity}:{d.code}:{d.message}" for d in parsed.diagnostics],
            "decode_error": decode_err,
            "usage": json.loads(json.dumps(getattr(resp, "usage", None), default=str)),
        }
        if parsed.masks and gt is not None:
            pred = masks_to_canvas(parsed, height, width)
            iou, dice = iou_dice(pred, gt)
            results["arm1_full_text"].update({"iou": iou, "dice": dice})
            print(f"decoded {len(parsed.boxes)} box / {len(parsed.masks)} mask records; "
                  f"single-image IoU={iou:.4f} Dice={dice:.4f}")
        elif parsed.masks:
            print(f"decoded {len(parsed.boxes)} box / {len(parsed.masks)} mask records; "
                  "no GT mask provided, IoU/Dice not computed.")
        else:
            print("decoded 0 mask records (empty or no masks).")
    except Exception as error:
        results["arm1_full_text"] = {"error": err_text(error)}
        print("API CALL FAILED:", err_text(error))

    # Step 2: box prompt probe (official wire format via build_object_prompt).
    print(f"\n--- [2] SAM {SAM_MODEL} BOX PROMPT PROBE (wire-format object prompt) ---")
    results["arm2_full_box"] = {}
    if box:
        wire_prompt = sam_decoder.build_box_prompt(
            int(box[0]), int(box[1]), int(box[2]), int(box[3]),
            width, height, convention=args.box_convention, object_id=1,
        )
        print("wire prompt:", wire_prompt)
        results["arm2_full_box"]["wire_prompt"] = wire_prompt
        try:
            resp = sam_call(client, image_data_url=data_url, prompt_text=wire_prompt,
                            include_confidence=args.include_confidence, retries=args.retries)
            text = getattr(resp, "output_text", None)
            print(f"resp.id={getattr(resp, 'id', None)} status={getattr(resp, 'status', None)}")
            print_verbatim("RAW resp.output_text", text)
            parsed, decode_err = decode_sam_response(resp)
            accepted = bool(parsed.boxes or parsed.masks)
            verdict = "ACCEPTED" if accepted else "REJECTED (empty records)"
            print(f"BOX PROMPT VERDICT: {verdict} — {len(parsed.boxes)} box, "
                  f"{len(parsed.masks)} mask records; diagnostics={len(parsed.diagnostics)}")
            results["arm2_full_box"].update({
                "response_id": getattr(resp, "id", None),
                "raw_output_text": text,
                "n_box_records": len(parsed.boxes),
                "n_mask_records": len(parsed.masks),
                "diagnostics": [f"{d.severity}:{d.code}:{d.message}" for d in parsed.diagnostics],
                "decode_error": decode_err,
                "verdict": verdict,
            })
            if parsed.masks and gt is not None:
                pred = masks_to_canvas(parsed, height, width)
                iou, dice = iou_dice(pred, gt)
                results["arm2_full_box"].update({"iou": iou, "dice": dice})
                print(f"single-image IoU={iou:.4f} Dice={dice:.4f}")
            elif parsed.masks:
                print("masks decoded but no GT mask provided; IoU/Dice not computed.")
        except Exception as error:
            results["arm2_full_box"] = {"wire_prompt": wire_prompt, "error": err_text(error),
                                        "verdict": "REJECTED (API error)"}
            print("BOX PROMPT REJECTED BY API:", err_text(error))
    else:
        results["arm2_full_box"] = {"skipped": "no box in manifest entry"}
        print("SKIPPED: no box available.")

    # Step 3: cropped-ROI text prompt (physical box fallback).
    print(f"\n--- [3] SAM {SAM_MODEL} CROPPED-ROI TEXT PROMPT (text='defect') ---")
    results["arm3_cropped_text"] = {}
    if box:
        cx1, cy1, cx2, cy2, crop_source = crop_box(item, width, height, args)
        print(f"crop ({crop_source}, half-open, clamped): ({cx1},{cy1})-({cx2},{cy2}) "
              f"size={cx2-cx1}x{cy2-cy1}")
        with Image.open(item["image_path"]) as im:
            crop = im.crop((cx1, cy1, cx2, cy2))
            crop_path = OUT_DIR / "smoke_crop.png"
            crop.save(crop_path)
        crop_url, (cw, ch) = image_to_data_url(crop_path)
        results["arm3_cropped_text"]["crop_source"] = crop_source
        results["arm3_cropped_text"]["crop_box_half_open"] = [cx1, cy1, cx2, cy2]
        try:
            resp = sam_call(client, image_data_url=crop_url, prompt_text="defect",
                            include_confidence=args.include_confidence, retries=args.retries)
            text = getattr(resp, "output_text", None)
            print(f"resp.id={getattr(resp, 'id', None)} status={getattr(resp, 'status', None)}")
            print_verbatim("RAW resp.output_text", text)
            parsed, decode_err = decode_sam_response(resp)
            results["arm3_cropped_text"].update({
                "response_id": getattr(resp, "id", None),
                "raw_output_text": text,
                "n_box_records": len(parsed.boxes),
                "n_mask_records": len(parsed.masks),
                "diagnostics": [f"{d.severity}:{d.code}:{d.message}" for d in parsed.diagnostics],
                "decode_error": decode_err,
            })
            if parsed.masks and gt is not None:
                # Paste crop-space masks back into the full canvas.
                crop_h, crop_w = cy2 - cy1, cx2 - cx1
                full = np.zeros((height, width), dtype=np.uint8)
                for mask in parsed.masks:
                    crop_canvas = sam_decoder.mask_to_full_canvas(mask, crop_h, crop_w)
                    full[cy1:cy2, cx1:cx2] = np.logical_or(
                        full[cy1:cy2, cx1:cx2], crop_canvas
                    ).astype(np.uint8)
                iou, dice = iou_dice(full, gt)
                results["arm3_cropped_text"].update({"iou": iou, "dice": dice})
                print(f"paste-back single-image IoU={iou:.4f} Dice={dice:.4f}")
            elif parsed.masks:
                print("masks decoded but no GT mask provided; IoU/Dice not computed.")
            else:
                print("decoded 0 mask records.")
        except Exception as error:
            results["arm3_cropped_text"]["error"] = err_text(error)
            print("API CALL FAILED:", err_text(error))
    else:
        results["arm3_cropped_text"] = {"skipped": "no box in manifest entry"}
        print("SKIPPED: no box available.")

    # Step 4: muse-spark-1.3 image + tool calling.
    print(f"\n--- [4] {MUSE_MODEL} IMAGE + TOOL CALLING VERIFICATION ---")
    results["muse_tool_calling"] = {}
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text",
             "text": "Inspect this industrial part image for surface defects. "
                     "Call record_defect_triage with your verdict."},
        ],
    }]
    for tool_choice in ("auto", "required"):
        label = f"tool_choice={tool_choice}"
        print(f"\n[muse attempt {label}]")
        try:
            resp = with_retries(
                lambda: client.chat.completions.create(
                    model=MUSE_MODEL, messages=messages, tools=[MUSE_TOOL],
                    tool_choice=tool_choice, temperature=0,
                ),
                what=f"muse:{tool_choice}", retries=args.retries,
            )
            message = resp.choices[0].message
            tool_calls = [tc.model_dump() for tc in (message.tool_calls or [])]
            print(f"resp.id={resp.id} finish_reason={resp.choices[0].finish_reason}")
            print(f"tool_calls: {json.dumps(tool_calls, indent=2) if tool_calls else '<None>'}")
            content = message.content
            print_verbatim("message.content", content, cap=1200)
            results["muse_tool_calling"][tool_choice] = {
                "response_id": resp.id,
                "finish_reason": resp.choices[0].finish_reason,
                "tool_calls": tool_calls,
                "content": content,
                "usage": json.loads(json.dumps(resp.usage, default=str)) if resp.usage else None,
            }
            if tool_calls:
                results["muse_tool_calling"]["verdict"] = (
                    f"VALID_TOOL_CALLS with tool_choice={tool_choice}"
                )
                break
        except Exception as error:
            results["muse_tool_calling"][tool_choice] = {"error": err_text(error)}
            print(f"MUSE CALL FAILED ({label}):", err_text(error))
    if "verdict" not in results["muse_tool_calling"]:
        results["muse_tool_calling"]["verdict"] = (
            "NO_TOOL_CALLS_RETURNED — model answered in content (or errored); "
            "see attempts above. Fallback: structured JSON output may be required."
        )
    print(f"\nMUSE VERDICT: {results['muse_tool_calling']['verdict']}")

    out_path = OUT_DIR / "smoke_result.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved raw smoke results to {out_path}")

    # Canonical flat schema required by scripts/gate0_report.py. PASS means the
    # live API call completed without error (an empty zero-match output_text is
    # a PASS with 0 masks, not a failure); acceptance of the box prompt is
    # reported separately via sam_box_accepted.
    def _status(arm: dict) -> str:
        return "FAIL" if arm.get("error") else "PASS"

    arm1 = results.get("arm1_full_text", {})
    arm2 = results.get("arm2_full_box", {})
    arm3 = results.get("arm3_cropped_text", {})
    muse = results.get("muse_tool_calling", {})
    # Consult the auto attempt first, then the required fallback; an auto dict
    # with no tool_calls (or an error dict) must not short-circuit the fallback.
    tool_calls = ((muse.get("auto") or {}).get("tool_calls")
                  or (muse.get("required") or {}).get("tool_calls") or [])
    tool_args: dict = {}
    if tool_calls:
        try:
            tool_args = json.loads(tool_calls[0]["function"]["arguments"])
        except Exception:
            tool_args = {}
    canonical = {
        "sam_text_status": _status(arm1),
        "sam_text_masks_found": int(arm1.get("n_mask_records", 0)),
        "sam_text_raw": str(arm1.get("raw_output_text") or "")[:300],
        "sam_box_status": "PASS" if (not arm2.get("error") and arm2.get("verdict") == "ACCEPTED") else "FAIL",
        "sam_box_accepted": bool(arm2.get("verdict") == "ACCEPTED"),
        "sam_box_raw": str(arm2.get("raw_output_text") or "")[:300],
        "sam_crop_status": _status(arm3),
        "sam_crop_masks_found": int(arm3.get("n_mask_records", 0)),
        "muse_spark_tool_status": "PASS" if tool_calls else "FAIL",
        "muse_spark_tool_name": tool_calls[0]["function"]["name"] if tool_calls else None,
        "muse_spark_tool_args": tool_args,
    }
    canon_path = OUT_DIR / "api_smoke_results.json"
    canon_path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved canonical audit schema to {canon_path}")
    return results


# ---------------------------------------------------------------------------
# Benchmark mode (20 images)
# ---------------------------------------------------------------------------

def predict_arm1(client, data_url, width, height, args):
    resp = sam_call(client, image_data_url=data_url, prompt_text="defect",
                    include_confidence=args.include_confidence, retries=args.retries)
    parsed, decode_err = decode_sam_response(resp)
    return resp, parsed, decode_err


def run_clahe_enhance(crop: Image.Image, out_path: Path) -> None:
    """CLAHE (clipLimit=2.0, 8x8 tiles) on the grayscale crop, saved back as RGB.

    Tests whether local contrast enhancement helps sam-3.1 find low-contrast
    SEM defect edges that the plain crop's zero-match responses miss.
    """
    import cv2

    gray = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    Image.fromarray(enhanced).convert("RGB").save(out_path)


def run_crop_arms(client: openai.OpenAI, item: dict, index: int, *, width: int,
                  height: int, gt: np.ndarray, args) -> dict:
    """Run crop_roi (plain) and crop_clahe (CLAHE-enhanced) on the same padded crop."""
    cx1, cy1, cx2, cy2, crop_source = crop_box(item, width, height, args)
    with Image.open(item["image_path"]) as im:
        crop = im.crop((cx1, cy1, cx2, cy2))
    crop_files = {}
    plain_path = OUT_DIR / f"crop_{index:02d}_roi.png"
    crop.save(plain_path)
    crop_files["crop_roi"] = plain_path
    clahe_path = OUT_DIR / f"crop_{index:02d}_clahe.png"
    run_clahe_enhance(crop, clahe_path)
    crop_files["crop_clahe"] = clahe_path

    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    results: dict = {}
    for arm, path in crop_files.items():
        entry: dict = {"crop_source": crop_source,
                       "crop_box_half_open": [cx1, cy1, cx2, cy2]}
        try:
            crop_url, _ = image_to_data_url(path)
            resp = sam_call(client, image_data_url=crop_url, prompt_text="defect",
                            include_confidence=args.include_confidence, retries=args.retries)
            parsed, decode_err = decode_sam_response(resp)
            entry.update({"response_id": getattr(resp, "id", None),
                          "raw_output_text": getattr(resp, "output_text", None),
                          "n_box_records": len(parsed.boxes),
                          "n_mask_records": len(parsed.masks),
                          "decode_error": decode_err})
            if parsed.masks:
                full = np.zeros((height, width), dtype=np.uint8)
                for mask in parsed.masks:
                    crop_canvas = sam_decoder.mask_to_full_canvas(mask, crop_h, crop_w)
                    full[cy1:cy2, cx1:cx2] = np.logical_or(
                        full[cy1:cy2, cx1:cx2], crop_canvas
                    ).astype(np.uint8)
                iou, dice = iou_dice(full, gt)
                entry.update({"iou": iou, "dice": dice})
        except Exception as error:
            entry["error"] = err_text(error)
        results[arm] = entry
    return results


def run_benchmark(client: openai.OpenAI, items: list[dict], args) -> dict:
    """20-image benchmark over four arms; writes the canonical audit schema.

    Arms: text_full (full image + "defect"), box_full (full image + official
    wire-format box prompt), crop_roi (padded crop + "defect"), crop_clahe
    (same crop, CLAHE contrast-enhanced + "defect").
    """
    arms_all = ("text_full", "box_full", "crop_roi", "crop_clahe")
    rows = []
    for index, item in enumerate(items):
        data_url, (width, height) = image_to_data_url(item["image_path"])
        gt = load_gt_mask(item["mask_path"])
        box = item["box"]
        row = {
            "sample_id": item["id"],
            "class_id": item["class_id"],
            "split_group": "nominal" if item["class_id"] in NOMINAL_CLASSES
                           else "rare" if item["class_id"] in RARE_CLASSES else "other",
            "image_size": [width, height],
            "gt_mask_pixels": int(gt.sum()),
        }
        details: dict = {}

        # Arm text_full: full-image text prompt.
        entry: dict = {}
        try:
            resp, parsed, decode_err = predict_arm1(client, data_url, width, height, args)
            entry = {"response_id": getattr(resp, "id", None),
                     "raw_output_text": getattr(resp, "output_text", None),
                     "n_box_records": len(parsed.boxes), "n_mask_records": len(parsed.masks),
                     "decode_error": decode_err}
            if parsed.masks:
                pred = masks_to_canvas(parsed, height, width)
                iou, dice = iou_dice(pred, gt)
                entry.update({"iou": iou, "dice": dice})
        except Exception as error:
            entry["error"] = err_text(error)
        details["text_full"] = entry

        # Arm box_full: full-image box prompt (only if a box exists).
        if box:
            entry = {}
            try:
                wire_prompt = sam_decoder.build_box_prompt(
                    int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                    width, height, convention=args.box_convention, object_id=1,
                )
                resp = sam_call(client, image_data_url=data_url, prompt_text=wire_prompt,
                                include_confidence=args.include_confidence, retries=args.retries)
                parsed, decode_err = decode_sam_response(resp)
                entry = {"wire_prompt": wire_prompt,
                         "response_id": getattr(resp, "id", None),
                         "raw_output_text": getattr(resp, "output_text", None),
                         "n_box_records": len(parsed.boxes), "n_mask_records": len(parsed.masks),
                         "decode_error": decode_err}
                if parsed.masks:
                    pred = masks_to_canvas(parsed, height, width)
                    iou, dice = iou_dice(pred, gt)
                    entry.update({"iou": iou, "dice": dice})
            except Exception as error:
                entry["error"] = err_text(error)
            details["box_full"] = entry
        else:
            details["box_full"] = {"skipped": "no box"}

        # Arms crop_roi + crop_clahe (padded crop, plain vs CLAHE-enhanced).
        if box:
            details.update(run_crop_arms(client, item, index, width=width, height=height,
                                         gt=gt, args=args))
        else:
            details["crop_roi"] = {"skipped": "no box"}
            details["crop_clahe"] = {"skipped": "no box"}

        row["details"] = details
        for arm in arms_all:
            value = details.get(arm, {})
            row[f"iou_{arm}"] = value.get("iou") if isinstance(value, dict) else None
        rows.append(row)
        ious = " ".join(f"{arm}={row[f'iou_{arm}'] if row[f'iou_{arm}'] is not None else 'ERR/0'}"
                        for arm in arms_all)
        print(f"[{index + 1}/{len(items)}] {item['id'][:12]} class={item['class_id']} {ious}")

    def _stats(group_rows: list[dict], arm: str) -> dict:
        ious, dices, errors = [], [], []
        for r in group_rows:
            e = r["details"].get(arm, {})
            if not isinstance(e, dict):
                continue
            if e.get("error"):
                errors.append(e["error"])
            if isinstance(e.get("iou"), (int, float)):
                ious.append(float(e["iou"]))
                if isinstance(e.get("dice"), (int, float)):
                    dices.append(float(e["dice"]))
        return {
            "n_images": len(group_rows),
            "n_scored": len(ious),
            "n_errors": len(errors),
            "median_iou": statistics.median(ious) if ious else None,
            "mean_iou": statistics.fmean(ious) if ious else None,
            "mean_dice": statistics.fmean(dices) if dices else None,
            "errors": errors,
        }

    summary = {}
    for arm in arms_all:
        overall = _stats(rows, arm)
        nominal = _stats([r for r in rows if r["split_group"] == "nominal"], arm)
        rare = _stats([r for r in rows if r["split_group"] == "rare"], arm)
        summary[arm] = {
            "median_iou": overall["median_iou"],
            "mean_iou": overall["mean_iou"],
            "mean_dice": overall["mean_dice"],
            "nominal_median_iou": nominal["median_iou"],
            "rare_median_iou": rare["median_iou"],
            "n_images": overall["n_images"],
            "n_scored": overall["n_scored"],
            "n_errors": overall["n_errors"],
            "nominal_n_scored": nominal["n_scored"],
            "rare_n_scored": rare["n_scored"],
            "nominal_mean_iou": nominal["mean_iou"],
            "rare_mean_iou": rare["mean_iou"],
            "errors": overall["errors"],
        }

    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY (Carinthia-S, 20 images, sam-3.1)")
    print("=" * 78)
    for arm, stats in summary.items():
        median_iou = "n/a" if stats["median_iou"] is None else f"{stats['median_iou']:.4f}"
        mean_iou = "n/a" if stats["mean_iou"] is None else f"{stats['mean_iou']:.4f}"
        nom_med = "n/a" if stats["nominal_median_iou"] is None else f"{stats['nominal_median_iou']:.4f}"
        rare_med = "n/a" if stats["rare_median_iou"] is None else f"{stats['rare_median_iou']:.4f}"
        print(f"\n{arm}: scored={stats['n_scored']}/{stats['n_images']} errors={stats['n_errors']}")
        print(f"  medianIoU={median_iou}  meanIoU={mean_iou}  meanDice="
              f"{'n/a' if stats['mean_dice'] is None else format(stats['mean_dice'], '.4f')}")
        print(f"  nominal(n={stats['nominal_n_scored']}) medianIoU={nom_med}  "
              f"rare(n={stats['rare_n_scored']}) medianIoU={rare_med}")

    document = {
        "benchmark": "gate0_carinthia_s_20",
        "model": SAM_MODEL,
        "api_base_url": API_BASE_URL,
        "box_convention": args.box_convention,
        "gt_bin_threshold": ">0 (matches manifest foreground_pixel_count on 20/20)",
        "include_confidence": bool(args.include_confidence),
        "n_items": len(rows),
        "summary": summary,
        "per_image": [
            {key: row[key] for key in ("sample_id", "class_id", "split_group",
                                       "iou_text_full", "iou_box_full",
                                       "iou_crop_roi", "iou_crop_clahe")}
            for row in rows
        ],
        "details": rows,
    }
    out_path = OUT_DIR / "sam_20_results.json"
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved canonical per-image table to {out_path}")
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["smoke", "benchmark20"])
    parser.add_argument("--manifest", type=Path, default=OUT_DIR / "gate0_20_manifest.json")
    parser.add_argument("--image", type=Path, help="single image for smoke (no manifest)")
    parser.add_argument("--mask", type=Path, help="ground-truth mask for --image")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--box", type=lambda s: [float(v) for v in s.split(",")],
                        help="x1,y1,x2,y2 half-open (or per --box-convention)")
    parser.add_argument("--box-convention", choices=["half_open", "inclusive"],
                        default="inclusive",
                        help="convention of manifest/GT boxes; the wire prompt always "
                             "carries inclusive x2/y2. Default 'inclusive': verified "
                             "against Agent 1's manifest (tight_bbox_xyxy matches the "
                             "GT mask bbox under mask>0 with inclusive x2/y2, 20/20)")
    parser.add_argument("--include-confidence", action="store_true",
                        help="send metadata include_confidence=true")
    parser.add_argument("--retries", type=int, default=2,
                        help="retries for 429/5xx/connection errors only (default 2)")
    parser.add_argument("--limit", type=int, default=None, help="limit benchmark images")
    parser.add_argument("--make-fixture", action="store_true",
                        help="generate a synthetic smoke fixture instead of failing "
                             "when manifest/image is unavailable")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = resolve_api_key()
    client = make_client(api_key)
    print(f"endpoint: {API_BASE_URL}  key: ***{api_key[-4:]}")

    if args.mode == "smoke":
        if args.image:
            if not args.image.is_file():
                raise SystemExit(f"FATAL: --image not found: {args.image}")
            mask = args.mask
            if mask and not mask.is_file():
                raise SystemExit(f"FATAL: --mask not found: {mask}")
            item = {"id": args.image.stem, "image_path": args.image.resolve(),
                    "mask_path": mask.resolve() if mask else args.image.resolve(),
                    "class_id": args.class_id, "box": args.box,
                    "has_gt": mask is not None}
            if not mask:
                print("WARNING: no GT mask provided; IoU will NOT be computed "
                      "(the image itself is never used as ground truth).")
        elif args.manifest.is_file():
            items = load_manifest(args.manifest)
            print(f"loaded {len(items)} manifest entries from {args.manifest}")
            item = items[0]
        elif args.make_fixture:
            print("No manifest and no --image; creating SYNTHETIC smoke fixture "
                  "(test input only — API responses remain real).")
            item = make_smoke_fixture(OUT_DIR)
        else:
            raise SystemExit(
                f"FATAL: no smoke input. {args.manifest} does not exist yet and no "
                "--image/--mask was given. Re-run with --make-fixture to use a "
                "clearly-labeled synthetic test image while waiting for Agent 1."
            )
        run_smoke(client, item, args)
    else:
        if not args.manifest.is_file():
            raise SystemExit(
                f"FATAL: {args.manifest} not found (Coding Agent 1 has not finished "
                "building it). Run --mode smoke in the meantime."
            )
        items = load_manifest(args.manifest)
        if args.limit:
            items = items[: args.limit]
        print(f"loaded {len(items)} manifest entries from {args.manifest}")
        run_benchmark(client, items, args)


if __name__ == "__main__":
    main()
