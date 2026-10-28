#!/usr/bin/env python3
"""Gate 0 follow-up diagnostic: cleaned boxes on low-IoU failures + symmetric jitter.

Part A: forensic decomposition of the 6 box_full failures (IoU < 0.50) — stray
right-border pixels (x >= 477, the documented 3px black-border artifact) vs
soft-mask halo (0 < mask < 128) — then live sam-3.1 re-calls with 3 cleaned
inclusive boxes per image, scored against both gt>0 and gt>=128, plus 4-panel
overlay PNGs.

Part B: symmetric per-edge jitter (uniform ±15%, 2 trials per image, seed 42;
corner images additionally jittered around their box_lcc_128) to test whether
the 0.8572 oracle-box median survives imperfect detector boxes.

File ownership (multi-agent rules):
  reads  data/carinthia_s/, results/gate0/{gate0_20_manifest,sam_20_results}.json
  writes results/gate0/diagnostic_jitter_and_cleaned_boxes.json
         outputs/gate0/failure_overlays/*.png (gitignored)
  never  modifies existing results/gate0/*.json (SHA-256 verified before/after)

Run: set -a; . ./.env; set +a
     .venv/Scripts/python.exe scripts/diagnose_failures_and_jitter.py [--skip-api]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_gate0_api_tests import (  # noqa: E402
    API_BASE_URL,
    SAM_MODEL,
    decode_sam_response,
    iou_dice,
    image_to_data_url,
    make_client,
    masks_to_canvas,
    resolve_api_key,
    sam_call,
)
from src.api import sam_decoder  # noqa: E402

RESULT_DIR = PROJECT_ROOT / "results" / "gate0"
MANIFEST_JSON = RESULT_DIR / "gate0_20_manifest.json"
BENCH_JSON = RESULT_DIR / "sam_20_results.json"
DIAG_JSON = RESULT_DIR / "diagnostic_jitter_and_cleaned_boxes.json"
OVERLAY_DIR = PROJECT_ROOT / "outputs" / "gate0" / "failure_overlays"
HASHED_FILES = ("sam_20_results.json", "gate0_20_manifest.json", "smoke_result.json",
                "dataset_ground_truth.json", "api_smoke_results.json")
BORDER_X = 477  # first column of the documented 3px right black border (480px wide)
W = H = 480


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def tight_bbox_inclusive(binary: np.ndarray) -> list[int] | None:
    """Inclusive [x1, y1, x2, y2] of nonzero pixels, or None when empty."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def largest_component(binary: np.ndarray) -> np.ndarray:
    """Largest 8-connected component of a binary mask (empty in, empty out)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return (labels == 1 + int(np.argmax(areas))).astype(np.uint8)


def cc_summary(binary: np.ndarray) -> dict:
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    areas = sorted((int(a) for a in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    return {"n_components": len(areas), "top5_areas_px": areas[:5]}


def iou_pair(pred: np.ndarray, gt_gt0: np.ndarray, gt_128: np.ndarray) -> dict:
    iou0, dice0 = iou_dice(pred, gt_gt0)
    iou128, dice128 = iou_dice(pred, gt_128)
    return {"iou_gt0": iou0, "dice_gt0": dice0, "iou_128": iou128, "dice_128": dice128}


# ---------------------------------------------------------------------------
# Overlay rendering (PIL only, 2x2 grid of 480x480 panels)
# ---------------------------------------------------------------------------

def render_overlay(image_rgb: Image.Image, gt_gt0: np.ndarray, gt_128: np.ndarray,
                   original_box: list[int], cleaned_box: list[int] | None,
                   sam_original: np.ndarray, sam_cleaned: np.ndarray,
                   caption_clean: str) -> Image.Image:
    size = image_rgb.size

    # Panel 1: SEM image + original oracle box (yellow) + cleaned LCC128 box (cyan).
    p1 = image_rgb.convert("RGB").copy()
    d = ImageDraw.Draw(p1)
    d.rectangle(original_box, outline=(255, 255, 0), width=2)
    if cleaned_box:
        d.rectangle(cleaned_box, outline=(0, 255, 255), width=2)

    # Panel 2: gt_gt0 in green with the gt_128 core in bright white on top.
    p2 = Image.fromarray(np.stack([
        np.where(gt_128, 255, 0),
        np.where(gt_128, 255, np.where(gt_gt0, 255, 0)),
        np.where(gt_128, 255, 0),
    ], axis=-1).astype(np.uint8), "RGB")

    # Panels 3/4: SAM masks in red / blue on black.
    def mask_panel(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        canvas[mask > 0] = color
        return Image.fromarray(canvas, "RGB")

    p3 = mask_panel(sam_original, (255, 0, 0))
    p4 = mask_panel(sam_cleaned, (0, 96, 255))

    def captioned(panel: Image.Image, text: str) -> Image.Image:
        out = panel.copy()
        ImageDraw.Draw(out).rectangle([0, 0, size[0] - 1, 14], fill=(0, 0, 0))
        ImageDraw.Draw(out).text((3, 2), text, fill=(255, 255, 255))
        return out

    grid = Image.new("RGB", (size[0] * 2, size[1] * 2), (20, 20, 20))
    grid.paste(captioned(p1, "SEM + oracle box (yellow) + cleaned LCC128 (cyan)"), (0, 0))
    grid.paste(captioned(p2, "GT mask>0 (green), mask>=128 core (white)"), (size[0], 0))
    grid.paste(captioned(p3, "original box_full SAM mask (red)"), (0, size[1]))
    grid.paste(captioned(p4, f"cleaned-box SAM mask (blue): {caption_clean}"), (size[0], size[1]))
    return grid


# ---------------------------------------------------------------------------
# Part A: forensics + cleaned-box calls on the low-IoU images
# ---------------------------------------------------------------------------

def run_part_a(client, manifest: list[dict], rows: list[dict], skip_api: bool) -> dict:
    by_id = {e["id"]: e for e in manifest}
    low = [r for r in rows
           if isinstance(r.get("iou_box_full"), (int, float)) and r["iou_box_full"] < 0.50]
    print(f"Part A: {len(low)} images with box_full IoU < 0.50: "
          f"{[r['sample_id'][:8] for r in low]}")
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    images_out = []
    lcc128_boxes: dict[str, list[int] | None] = {}
    for row in low:
        entry = by_id[row["sample_id"]]
        img_id = entry["id"]
        cls = entry["class_label"]
        raw = np.array(Image.open(entry["original_mask_path"]).convert("L"))
        gt_gt0 = raw > 0
        gt_128 = raw >= 128

        # Forensics: border pixels, connected components (border zeroed), SAM mask.
        no_border = raw.copy()
        no_border[:, BORDER_X:] = 0
        nb_gt0 = no_border > 0
        nb_128 = no_border >= 128
        forensics = {
            "gt_gt0_px": int(gt_gt0.sum()),
            "gt_128_px": int(gt_128.sum()),
            "right_border_gt0_px": int(gt_gt0[:, BORDER_X:].sum()),
            "right_border_128_px": int(gt_128[:, BORDER_X:].sum()),
            "top_border_gt0_px": int(gt_gt0[0, :].sum()),
            "top_border_128_px": int(gt_128[0, :].sum()),
            "cc_gt0_after_border_zero": cc_summary(nb_gt0),
            "cc_128_after_border_zero": cc_summary(nb_128),
            "tight_bbox_gt0_inclusive": tight_bbox_inclusive(gt_gt0),
            "tight_bbox_128_inclusive": tight_bbox_inclusive(gt_128),
        }
        # Stored original box_full SAM mask, decoded from the retained raw text.
        stored_text = row["details"]["box_full"].get("raw_output_text") or ""
        if stored_text:
            parsed_stored, _err = decode_sam_response(
                type("StoredResponse", (), {"output_text": stored_text})()
            )
            sam_original = masks_to_canvas(parsed_stored, H, W)
            sam_original_box = None
            if parsed_stored.boxes:
                b = parsed_stored.boxes[0]
                sam_original_box = [b.left, b.top, b.right - 1, b.bottom - 1]
        else:
            sam_original, sam_original_box = np.zeros((H, W), dtype=np.uint8), None
        forensics["stored_sam_mask_px"] = int(sam_original.sum())
        forensics["stored_sam_bbox_inclusive"] = sam_original_box
        forensics["stored_sam_fill_ratio"] = float(sam_original.sum() / (W * H))

        # Three cleaned boxes (inclusive), border zeroed before each computation.
        lcc_gt0 = largest_component(nb_gt0)
        lcc_128 = largest_component(nb_128)
        cleaned = {
            "box_thr128": tight_bbox_inclusive(nb_128),
            "box_lcc_gt0": tight_bbox_inclusive(lcc_gt0),
            "box_lcc_128": tight_bbox_inclusive(lcc_128),
        }
        lcc128_boxes[img_id] = cleaned["box_lcc_128"]

        results = {}
        sam_cleaned_canvas = np.zeros((H, W), dtype=np.uint8)
        caption_clean = "none (empty)"
        for name, box in cleaned.items():
            if box is None:
                results[name] = {"box_inclusive": None, "skipped": "empty after cleaning"}
                continue
            entry_out = {"box_inclusive": box}
            if not skip_api:
                data_url, _ = image_to_data_url(Path(entry["rgb_png_path"]))
                prompt = sam_decoder.build_box_prompt(
                    box[0], box[1], box[2], box[3], W, H,
                    convention="inclusive", object_id=1,
                )
                resp = sam_call(client, image_data_url=data_url, prompt_text=prompt, retries=2)
                parsed, decode_err = decode_sam_response(resp)
                entry_out.update({
                    "response_id": getattr(resp, "id", None),
                    "raw_output_text": getattr(resp, "output_text", None),
                    "n_mask_records": len(parsed.masks),
                    "decode_error": decode_err,
                })
                if parsed.masks:
                    pred = masks_to_canvas(parsed, H, W)
                    entry_out.update(iou_pair(pred, gt_gt0, gt_128))
                    entry_out["sam_mask_px"] = int(pred.sum())
                    if name == "box_lcc_128" or (not sam_cleaned_canvas.any()
                                                 and name == "box_lcc_gt0"):
                        sam_cleaned_canvas = pred
                        caption_clean = name
            results[name] = entry_out

        overlay = render_overlay(
            Image.open(entry["rgb_png_path"]), gt_gt0, gt_128,
            entry["tight_bbox_xyxy"], cleaned["box_lcc_128"],
            sam_original, sam_cleaned_canvas, caption_clean,
        )
        overlay_path = OVERLAY_DIR / f"class{cls}_{img_id[:8]}.png"
        overlay.save(overlay_path)

        print(f"  {img_id[:8]} class={cls} original_iou={row.get('iou_box_full'):.4f} | "
              + " | ".join(
                  f"{k}: box={v['box_inclusive']}"
                  + (f" iou0={v['iou_gt0']:.3f}/128={v['iou_128']:.3f}"
                     if isinstance(v.get("iou_gt0"), (int, float)) else " skipped")
                  for k, v in results.items()))
        images_out.append({
            "id": img_id, "class_id": cls,
            "original_iou_box_full": row.get("iou_box_full"),
            "forensics": forensics,
            "cleaned_boxes": cleaned,
            "cleaned_box_results": results,
            "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
        })
    return {"selection_criterion": "iou_box_full < 0.50", "images": images_out,
            "lcc128_boxes": lcc128_boxes}


# ---------------------------------------------------------------------------
# Part B: symmetric ±15% jitter (oracle boxes; corner images also around lcc128)
# ---------------------------------------------------------------------------

def jitter_box(rng, box: list[int]) -> list[int]:
    x1, y1, x2, y2 = box
    bw, bh = max(4, x2 - x1), max(4, y2 - y1)
    jx1 = int(np.clip(round(x1 + rng.uniform(-0.15, 0.15) * bw), 0, W - 3))
    jy1 = int(np.clip(round(y1 + rng.uniform(-0.15, 0.15) * bh), 0, H - 3))
    jx2 = int(np.clip(round(x2 + rng.uniform(-0.15, 0.15) * bw), jx1 + 2, W - 1))
    jy2 = int(np.clip(round(y2 + rng.uniform(-0.15, 0.15) * bh), jy1 + 2, H - 1))
    return [jx1, jy1, jx2, jy2]


def call_sam_box(client, data_url: str, box: list[int], gt_gt0: np.ndarray,
                 gt_128: np.ndarray) -> dict:
    prompt = sam_decoder.build_box_prompt(box[0], box[1], box[2], box[3], W, H,
                                          convention="inclusive", object_id=1)
    resp = sam_call(client, image_data_url=data_url, prompt_text=prompt, retries=2)
    parsed, decode_err = decode_sam_response(resp)
    record = {"response_id": getattr(resp, "id", None),
              "raw_output_text": getattr(resp, "output_text", None),
              "n_mask_records": len(parsed.masks),
              "decode_error": decode_err}
    if parsed.masks:
        pred = masks_to_canvas(parsed, H, W)
        record.update(iou_pair(pred, gt_gt0, gt_128))
    return record


def run_part_b(client, manifest: list[dict], rows: list[dict], skip_api: bool,
               corner_ids: set[str], clean_ids: set[str],
               lcc128_boxes: dict[str, list[int] | None]) -> dict:
    by_row = {r["sample_id"]: r for r in rows}

    def group_of(sample_id: str) -> str:
        if sample_id in clean_ids:
            return "clean14"
        if sample_id in corner_ids:
            return "corner5"
        return "class6_exception"

    rng = np.random.default_rng(42)
    trials_out = []
    for entry in manifest:
        sample_id = entry["id"]
        box = [int(v) for v in entry["tight_bbox_xyxy"]]
        raw = np.array(Image.open(entry["original_mask_path"]).convert("L"))
        gt_gt0, gt_128 = raw > 0, raw >= 128
        data_url, _ = image_to_data_url(Path(entry["rgb_png_path"]))
        base = {"id": sample_id, "class_id": entry["class_label"],
                "group": group_of(sample_id), "oracle_box_inclusive": box}

        for trial in (1, 2):
            jbox = jitter_box(rng, box)
            record = {**base, "prompt_source": "oracle_box_jitter", "trial": trial,
                      "jittered_box_inclusive": jbox}
            if not skip_api:
                record.update(call_sam_box(client, data_url, jbox, gt_gt0, gt_128))
            trials_out.append(record)
        if sample_id in corner_ids and lcc128_boxes.get(sample_id):
            for trial in (1, 2):
                jbox = jitter_box(rng, lcc128_boxes[sample_id])
                record = {**base, "prompt_source": "box_lcc_128_jitter", "trial": trial,
                          "lcc128_box": lcc128_boxes[sample_id],
                          "jittered_box_inclusive": jbox}
                if not skip_api:
                    record.update(call_sam_box(client, data_url, jbox, gt_gt0, gt_128))
                trials_out.append(record)
        mine = [t for t in trials_out if t["id"] == sample_id]
        pair_txt = " ".join(
            f"{t['prompt_source'][:5]}t{t['trial']}:"
            + (f"iou0={t['iou_gt0']:.3f}" if isinstance(t.get("iou_gt0"), (int, float)) else "no-mask")
            for t in mine
        )
        print(f"[{sum(1 for t in trials_out if t['prompt_source'] == 'oracle_box_jitter') // 2}/20] "
              f"{sample_id[:8]} group={group_of(sample_id)} {pair_txt}")

    def stats_for(group: str, key: str, source: str = "oracle_box_jitter") -> dict:
        vals = [t[key] for t in trials_out
                if (group == "all20" or t["group"] == group)
                and (source is None or t["prompt_source"] == source) and key in t]
        return {"n": len(vals), "median": statistics.median(vals) if vals else None,
                "mean": statistics.fmean(vals) if vals else None}

    groups = {}
    for g in ("clean14", "corner5", "class6_exception", "all20"):
        groups[g] = {key: stats_for(g, key, source="oracle_box_jitter")
                     for key in ("iou_gt0", "iou_128", "dice_gt0", "dice_128")}
    # Corner images re-centered on their cleaned LCC boxes, separately.
    groups["corner5_lcc128_jitter"] = {
        key: stats_for("corner5", key, source="box_lcc_128_jitter")
        for key in ("iou_gt0", "iou_128", "dice_gt0", "dice_128")
    }
    return {"n_images": len(manifest), "n_trials": len(trials_out),
            "group_counts": {g: len({t["id"] for t in trials_out if t["group"] == g})
                             for g in ("clean14", "corner5", "class6_exception")},
            "trials": trials_out, "groups": groups}


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-api", action="store_true",
                        help="offline forensics only (no sam-3.1 calls)")
    args = parser.parse_args()

    hashes_before = {name: sha256(RESULT_DIR / name) for name in HASHED_FILES}
    print("SHA-256 before:")
    for name, digest in hashes_before.items():
        print(f"  {name}: {digest[:16]}...")

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))["samples"]
    for e in manifest:  # manifest paths are relative to the project root
        for key in ("original_mask_path", "rgb_png_path", "original_image_path"):
            if key in e:
                p = Path(e[key])
                e[key] = str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())
        if not Path(e["rgb_png_path"]).is_file() and Path(e["original_image_path"]).is_file():
            e["rgb_png_path"] = e["original_image_path"]  # fallback per repo rules
    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    rows = bench["per_image"]
    for r in rows:
        r["details"] = next(d["details"] for d in bench["details"]
                            if d["sample_id"] == r["sample_id"])

    # Corner-image detection straight from the manifest (top-right touching boxes).
    corner_ids: set[str] = set()
    for r in rows:
        if isinstance(r.get("iou_box_full"), (int, float)) and r["iou_box_full"] < 0.50:
            e = next(e for e in manifest if e["id"] == r["sample_id"])
            _x1, y1, x2, _y2 = e["tight_bbox_xyxy"]
            if y1 == 0 and x2 >= BORDER_X:
                corner_ids.add(r["sample_id"])
    clean_ids = {r["sample_id"] for r in rows
                 if isinstance(r.get("iou_box_full"), (int, float)) and r["iou_box_full"] >= 0.50}
    print(f"groups: clean={len(clean_ids)} corner={len(corner_ids)} "
          f"other={len(rows) - len(clean_ids) - len(corner_ids)}")

    client = None
    if not args.skip_api:
        client = make_client(resolve_api_key())
        print(f"endpoint: {API_BASE_URL}  model: {SAM_MODEL}")

    part_a = run_part_a(client, manifest, rows, args.skip_api)
    part_b = run_part_b(client, manifest, rows, args.skip_api, corner_ids, clean_ids,
                        part_a["lcc128_boxes"])

    document = {
        "purpose": "Gate 0 follow-up: stray-pixel/soft-mask forensics + cleaned-box "
                   "re-calls + symmetric ±15% jitter sensitivity",
        "model": SAM_MODEL, "api_base_url": API_BASE_URL,
        "border_zero_column": BORDER_X,
        "jitter": "2 trials/image, per-edge uniform(-0.15,+0.15)*bw|bh, seed 42; "
                  "corner images additionally jittered around box_lcc_128",
        "sha256_of_results_before": hashes_before,
        "partA": part_a,
        "partB": part_b,
    }
    DIAG_JSON.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved diagnostic to {DIAG_JSON}")

    hashes_after = {name: sha256(RESULT_DIR / name) for name in HASHED_FILES}
    unchanged = hashes_after == hashes_before
    document["sha256_of_results_unchanged"] = unchanged
    DIAG_JSON.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"results/gate0/*.json unchanged after run: {unchanged}")
    if not unchanged:
        raise SystemExit("FATAL: an existing results/gate0 JSON was modified — investigate.")

    print("\n" + "=" * 78)
    print("PART B SUMMARY — symmetric ±15% jitter")
    print("=" * 78)
    for g, stats in part_b["groups"].items():
        def fmt(d: dict) -> str:
            med = "n/a" if d["median"] is None else f"{d['median']:.4f}"
            mean = "n/a" if d["mean"] is None else f"{d['mean']:.4f}"
            return f"n={d['n']:2d}  median={med}  mean={mean}"
        print(f"\n{g}:")
        print(f"  iou vs gt>0   : {fmt(stats['iou_gt0'])}")
        print(f"  iou vs gt>=128: {fmt(stats['iou_128'])}")
        print(f"  dice vs gt>0  : {fmt(stats['dice_gt0'])}")


if __name__ == "__main__":
    main()
