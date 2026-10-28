"""SAM 3.1 wire-format decoder for the Meta Model API (Gate 0).

Wraps the official reference implementation ``meta-sam-parser``
(``pip install meta-sam-parser``; source: https://github.com/meta-models/meta-sam)
which implements the "SAM 3 text format" defined in that repo's
``protocol/sam3.md``. Mask payload decoding (base85 + range/arithmetic coder)
is delegated to the official package verbatim, so bit-exactness with the API
is guaranteed by upstream rather than re-implemented here.

Wire grammar highlights (protocol/sam3.md):
  - ``output_text`` carries one line per frame:
    ``<Nf><object_id><|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;x=0;y=0;data=H,W,PAYLOAD|>``
  - Box ``x2``/``y2`` are INCLUSIVE source-media coordinates; normalized
    half-open edges are ``right = x2 + 1``, ``bottom = y2 + 1``.
  - Mask ``data`` is ``height,width,payload``. The payload's first character is
    the encoding marker (``!`` = one_bit, ``~`` = lossless); the rest is base85
    (NOT base64) digits over the 85-character alphabet that excludes the wire
    delimiters ``" \\ , ; < > |``.
  - Optional per-token detection confidence ``c`` is returned when the request
    sets Responses ``metadata: {"include_confidence": "true"}``.
  - Box prompts (request ``input_text``) reuse the output token vocabulary:
    ``<0f>1<|box;x1=L;y1=T;x2=R-1;y2=B-1;w=W;h=H|>`` built via
    ``meta_sam_parser.build_object_prompt``. Every token carries the media's
    displayed ``w``/``h``; coordinates must be non-negative and inside it.

Verification vectors (both are Meta-published ground truth, not synthetic):
  1. ``python/consumer/openai_smoke.py`` in the reference repo: the 5x5 one_bit
     mask ``!!!!!(QO(0lu8?`` decodes to raster ``1001000110000001011110001``.
  2. ``conformance/cases/captured-api-image-lossless.json``: a verbatim
     ``output_text`` captured from a live SAM 3 image response, whose 11x11
     lossless mask decodes to an exact 121-byte raster with COCO RLE counts
     ``4275L2O0O=0D0O101O2LN``.

Requires: ``pip install meta-sam-parser numpy`` (Python >= 3.12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import meta_sam_parser as _msp
    from meta_sam_parser import (
        CompletedOutcome,
        SegmentationBoxRecord,
        SegmentationDiagnostic,
        SegmentationMaskRecord,
        SegmentationTextRecord,
        build_object_prompt as _build_object_prompt,
        decode_mask_to_raster as _decode_mask_to_raster,
        decode_mask_to_rle as _decode_mask_to_rle,
        image_segmentation_format as _image_segmentation_format,
        video_segmentation_format as _video_segmentation_format,
    )
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "sam_decoder requires the official reference parser: "
        "pip install meta-sam-parser  (source: github.com/meta-models/meta-sam)"
    ) from error

__all__ = [
    "SamBox",
    "SamMaskRecord",
    "SamParsedResponse",
    "build_box_prompt",
    "build_point_prompt",
    "decode_output_text",
    "mask_to_full_canvas",
    "run_self_test",
]


@dataclass(frozen=True)
class SamBox:
    """Parsed box in half-open source-media coordinates (right/bottom exclusive)."""

    object_id: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float | None


@dataclass(frozen=True)
class SamMaskRecord:
    """One decoded mask, in mask-local raster space plus its full-canvas bounds."""

    object_id: str
    encoding: str  # "one_bit" | "lossless"
    payload: str
    width: int
    height: int
    bounds: tuple[int, int, int, int]  # (left, top, right, bottom), half-open
    confidence: float | None
    raster: np.ndarray  # uint8 (height, width), values {0, 1}


@dataclass(frozen=True)
class SamParsedResponse:
    """Structured result of parsing one SAM 3.1 ``output_text``."""

    boxes: tuple[SamBox, ...] = ()
    masks: tuple[SamMaskRecord, ...] = ()
    diagnostics: tuple[SegmentationDiagnostic, ...] = ()
    raw_output: str = ""
    prose: tuple[str, ...] = field(default=())


def decode_output_text(
    output_text: str, *, decode_rasters: bool = True, media: str = "image"
) -> SamParsedResponse:
    """Parse a complete (non-streaming) SAM 3.1 ``output_text``.

    Uses the official segmentation-format parser from ``meta-sam-parser``
    (``media="image"`` requires frame ``<0f>`` lines; ``media="video"`` keeps
    frame references); malformed lines become diagnostics, accepted records
    are returned in order.
    """
    format_factory = (
        _image_segmentation_format() if media == "image" else _video_segmentation_format()
    )
    parser = format_factory.create_parser()
    parser.push(output_text)
    result = parser.finish(CompletedOutcome()).result

    boxes: list[SamBox] = []
    masks: list[SamMaskRecord] = []
    prose: list[str] = []
    for record in result.records:
        if isinstance(record, SegmentationBoxRecord):
            boxes.append(
                SamBox(
                    object_id=record.object_id,
                    left=record.left,
                    top=record.top,
                    right=record.right,
                    bottom=record.bottom,
                    confidence=record.confidence,
                )
            )
        elif isinstance(record, SegmentationMaskRecord):
            raster = None
            if decode_rasters:
                raster = np.frombuffer(
                    _decode_mask_to_raster(record.mask), dtype=np.uint8
                ).reshape(record.mask.height, record.mask.width).copy()
            masks.append(
                SamMaskRecord(
                    object_id=record.object_id,
                    encoding=record.mask.encoding,
                    payload=record.mask.payload,
                    width=record.mask.width,
                    height=record.mask.height,
                    bounds=(
                        record.bounds.left,
                        record.bounds.top,
                        record.bounds.right,
                        record.bounds.bottom,
                    ),
                    confidence=record.confidence,
                    raster=raster,
                )
            )
        elif isinstance(record, SegmentationTextRecord):
            prose.append(record.text)

    return SamParsedResponse(
        boxes=tuple(boxes),
        masks=tuple(masks),
        diagnostics=tuple(result.diagnostics),
        raw_output=result.raw_output,
        prose=tuple(prose),
    )


def mask_to_full_canvas(
    record: SamMaskRecord, canvas_height: int, canvas_width: int
) -> np.ndarray:
    """Place a decoded mask record into a full-canvas binary raster (H, W).

    The mask is bounded by ``record.bounds`` in source-media coordinates; the
    rest of the canvas is zeros.
    """
    left, top, right, bottom = record.bounds
    if (right - left, bottom - top) != (record.width, record.height):
        raise ValueError(
            f"Mask raster {record.width}x{record.height} does not match its "
            f"bounds {right - left}x{bottom - top}."
        )
    if not (0 <= left < right <= canvas_width and 0 <= top < bottom <= canvas_height):
        raise ValueError(
            f"Mask bounds {record.bounds} fall outside the "
            f"{canvas_width}x{canvas_height} canvas."
        )
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas[top:bottom, left:right] = record.raster
    return canvas


def build_box_prompt(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    media_width: int,
    media_height: int,
    *,
    convention: str = "half_open",
    object_id: int | str = 1,
) -> str:
    """Build the official object-prompt ``input_text`` for one box.

    ``convention`` describes the caller's ``x2``/``y2``: ``"half_open"`` means
    exclusive right/bottom edges (converted to the wire's inclusive form via
    ``right - 1``); ``"inclusive"`` passes them through unchanged. The wire
    always carries inclusive ``x2``/``y2`` plus the media's ``w``/``h``.
    """
    if convention not in ("half_open", "inclusive"):
        raise ValueError(f"Unknown box convention: {convention!r}")
    if convention == "half_open":
        x2 -= 1
        y2 -= 1
    return _build_object_prompt(
        size=(media_width, media_height),
        objects=[
            _msp.PromptObject(
                id=object_id,
                prompts=[
                    _msp.FramePrompt(
                        frame=0,
                        box=_msp.PromptBox(left=x1, top=y1, right=x2 + 1, bottom=y2 + 1),
                    )
                ],
            )
        ],
    )


def build_point_prompt(
    x: int,
    y: int,
    media_width: int,
    media_height: int,
    *,
    label: str = "positive",
    object_id: int | str = 1,
) -> str:
    """Build the official object-prompt ``input_text`` for one point click."""
    return _build_object_prompt(
        size=(media_width, media_height),
        objects=[
            _msp.PromptObject(
                id=object_id,
                prompts=[
                    _msp.FramePrompt(
                        frame=0,
                        points=[_msp.PromptPoint(x=x, y=y, label=label)],
                    )
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Verification vectors (Meta-published ground truth)
# ---------------------------------------------------------------------------

_SMOKE_LINE = (
    "<7f>0<|box;x1=10;y1=20;x2=14;y2=24;w=200;h=100|>"
    "<|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>\n"
)
_SMOKE_RASTER = "1001000110000001011110001"
_SMOKE_RLE_COUNTS = "01214OK0010O31"

_CAPTURED_LINE = (
    "<0f>0<|box;x1=513;y1=371;x2=523;y2=381;w=1440;h=900|>"
    "<|mask;x=0;y=0;data=11,11,~!!!!r!kCq#$.pF[_cZIw.?wO3R8-&NJofiCI$_J0aA"
    ".1VC*YyK_'z*(LEG8!+H9evY.82Z-4.@C[.9C@T86o$3E=-&8w3W=]xS.|>\n"
)
_CAPTURED_RASTER = (
    "00001100000001111111000111111111001111111110111111111111111111111101111111"
    "11101111111111011111111100011111111000001110000"
)
_CAPTURED_RLE_COUNTS = "4275L2O0O=0D0O101O2LN"
_CAPTURED_BOUNDS = (513, 371, 524, 382)  # half-open, from the conformance case


def run_self_test(verbose: bool = True) -> bool:
    """Verify the decoder against the two Meta-published ground-truth vectors.

    Returns True when every assertion passes; raises AssertionError otherwise.
    """
    # Vector 1: official openai_smoke.py line (one_bit, video frame marker).
    parsed = decode_output_text(_SMOKE_LINE, media="video")
    assert len(parsed.boxes) == 1 and len(parsed.masks) == 1, parsed
    box = parsed.boxes[0]
    assert (box.left, box.top, box.right, box.bottom) == (10, 20, 15, 25), box
    raster = parsed.masks[0].raster.tobytes().decode("latin-1").replace("\x00", "0").replace("\x01", "1")
    assert parsed.masks[0].encoding == "one_bit"
    assert raster == _SMOKE_RASTER, f"one_bit raster mismatch: {raster}"
    rle = _decode_mask_to_rle(
        _msp.SegmentationMask("one_bit", parsed.masks[0].payload, 5, 5)
    )
    assert rle.counts == _SMOKE_RLE_COUNTS, rle.counts

    # Vector 2: verbatim live-API captured response (lossless).
    parsed = decode_output_text(_CAPTURED_LINE)
    assert len(parsed.boxes) == 1 and len(parsed.masks) == 1, parsed
    assert parsed.diagnostics == (), parsed.diagnostics
    mask = parsed.masks[0]
    assert mask.encoding == "lossless"
    assert mask.bounds == _CAPTURED_BOUNDS, mask.bounds
    raster = mask.raster.tobytes().decode("latin-1").replace("\x00", "0").replace("\x01", "1")
    assert raster == _CAPTURED_RASTER, f"lossless raster mismatch: {raster}"
    rle = _decode_mask_to_rle(
        _msp.SegmentationMask("lossless", mask.payload, 11, 11)
    )
    assert rle.counts == _CAPTURED_RLE_COUNTS, rle.counts
    canvas = mask_to_full_canvas(mask, canvas_height=900, canvas_width=1440)
    assert canvas.shape == (900, 1440)
    left, top, right, bottom = _CAPTURED_BOUNDS
    assert (canvas[top:bottom, left:right] == mask.raster).all()
    assert canvas.sum() == int(mask.raster.sum())  # nothing spilled outside bounds

    # Prompt builder round trip (official build_object_prompt output).
    prompt = build_box_prompt(513, 371, 524, 382, 1440, 900, convention="half_open")
    assert prompt == "<0f>1<|box;x1=513;y1=371;x2=523;y2=381;w=1440;h=900|>", prompt

    if verbose:
        print("sam_decoder self-test PASSED (2/2 ground-truth vectors + prompt builder)")
        print(f"  one_bit  5x5  raster: {raster[:25] and _SMOKE_RASTER}")
        print(f"  lossless 11x11 raster: {_CAPTURED_RASTER}")
        print(f"  box prompt built: {prompt}")
    return True


if __name__ == "__main__":
    run_self_test()
