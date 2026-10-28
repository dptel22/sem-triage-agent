"""SAM 3.1 wire-format decoder and API client for the Meta Model API.

Wraps the official reference implementation ``meta-sam-parser``
(``pip install meta-sam-parser``; source: https://github.com/meta-models/meta-sam)
which implements the "SAM 3 text format" defined in that repo's
``protocol/sam3.md``. Mask payload decoding (base85 + range/arithmetic coder)
is delegated to the official package verbatim, guaranteeing bit-exactness.
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
        image_segmentation_format as _image_segmentation_format,
        video_segmentation_format as _video_segmentation_format,
    )
    _META_SAM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _META_SAM_AVAILABLE = False


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
    """Parse a complete (non-streaming) SAM 3.1 ``output_text``."""
    if not _META_SAM_AVAILABLE:
        raise ImportError("meta-sam-parser is required to decode SAM outputs")

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
    record: SamMaskRecord, canvas_height: int = 480, canvas_width: int = 480
) -> np.ndarray:
    """Place a decoded mask record into a full-canvas binary raster (H, W)."""
    left, top, right, bottom = record.bounds
    if (right - left, bottom - top) != (record.width, record.height):
        raise ValueError(
            f"Mask raster {record.width}x{record.height} does not match its "
            f"bounds {right - left}x{bottom - top}."
        )
    if not (0 <= left < right <= canvas_width and 0 <= top < bottom <= canvas_height):
        raise ValueError(
            f"Mask bounds {record.bounds} fall outside the {canvas_width}x{canvas_height} canvas."
        )
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas[top:bottom, left:right] = record.raster
    return canvas


def build_box_prompt(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    canvas_w: int = 480,
    canvas_h: int = 480,
    convention: str = "inclusive",
    object_id: int = 1,
) -> str:
    """Format a single-box prompt string accepted by the SAM 3.1 Responses API."""
    if not _META_SAM_AVAILABLE:
        raise ImportError("meta-sam-parser is required to build prompts")

    if convention == "inclusive":
        right_inclusive = x2
        bottom_inclusive = y2
    elif convention == "half_open":
        right_inclusive = x2 - 1
        bottom_inclusive = y2 - 1
    else:
        raise ValueError(f"Unknown convention '{convention}'; use 'inclusive' or 'half_open'")

    prompt_box = _msp.PromptBox(
        left=int(x1),
        top=int(y1),
        right=int(right_inclusive) + 1,
        bottom=int(bottom_inclusive) + 1,
    )
    obj = _msp.PromptObject(
        id=object_id,
        prompts=[_msp.FramePrompt(frame=0, box=prompt_box)],
    )
    return _build_object_prompt(size=(canvas_w, canvas_h), objects=[obj])


def build_point_prompt(
    x: int,
    y: int,
    label: str = "positive",
    canvas_w: int = 480,
    canvas_h: int = 480,
    object_id: int = 1,
) -> str:
    """Format a single-point prompt string accepted by the SAM 3.1 Responses API."""
    if not _META_SAM_AVAILABLE:
        raise ImportError("meta-sam-parser is required to build prompts")

    pt = _msp.PromptPoint(x=int(x), y=int(y), label=label)
    obj = _msp.PromptObject(
        id=object_id,
        prompts=[_msp.FramePrompt(frame=0, points=[pt])],
    )
    return _build_object_prompt(size=(canvas_w, canvas_h), objects=[obj])


def run_self_test(verbose: bool = False) -> bool:
    """Run upstream verification vectors against the installed parser."""
    line_onebit = (
        "<0f>1"
        "<|box;x1=0;y1=0;x2=4;y2=4;w=5;h=5|>"
        "<|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>"
    )
    parsed = decode_output_text(line_onebit)
    if len(parsed.masks) != 1:
        return False
    rec = parsed.masks[0]
    expected_raster = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    return bool(np.array_equal(rec.raster, expected_raster))
