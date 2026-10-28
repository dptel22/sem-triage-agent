"""Deterministic gates and mask verification for Condition C3 and Condition C1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EmptyScanGateResult:
    is_empty: bool
    filtered_fg_pixels: int
    raw_fg_pixels: int
    reason: str


def evaluate_empty_scan_gate(
    unet_pred_entry: Dict[str, Any],
    threshold_pixels: int = 64,
) -> EmptyScanGateResult:
    """Evaluates Condition C3: Deterministic Empty-Scan Gate.

    Routes empty or misaligned SEM scans (class 6) directly to
    'NO_DEFECT / REIMAGE_REQUIRED' without calling expensive downstream APIs.

    Tuned on nominal_val (0/430 flagged, min fg = 255) vs class6_val (23/23 flagged).
    """
    filt_px = int(unet_pred_entry.get("predicted_foreground_pixels_filtered", 0))
    raw_px = int(unet_pred_entry.get("predicted_foreground_pixels_raw", 0))
    largest_box = unet_pred_entry.get("largest_component_bbox")

    if largest_box is None:
        return EmptyScanGateResult(
            is_empty=True,
            filtered_fg_pixels=filt_px,
            raw_fg_pixels=raw_px,
            reason="No connected foreground component survived the area >= 16px filter",
        )

    if filt_px < threshold_pixels:
        return EmptyScanGateResult(
            is_empty=True,
            filtered_fg_pixels=filt_px,
            raw_fg_pixels=raw_px,
            reason=f"Filtered foreground pixels ({filt_px}) < empty-scan threshold ({threshold_pixels})",
        )

    return EmptyScanGateResult(
        is_empty=False,
        filtered_fg_pixels=filt_px,
        raw_fg_pixels=raw_px,
        reason="Defect component detected with sufficient foreground mass",
    )


@dataclass(frozen=True)
class MaskVerificationResult:
    unet_sam_iou: float
    box_fill_ratio: float
    prompt_box_fill_ratio: float
    sam_bbox_fill_ratio: float
    area_ratio: float
    triggered_refinement: bool
    trigger_reasons: List[str]
    is_locked: bool


def verify_sam_mask(
    unet_mask: np.ndarray,
    sam_mask: np.ndarray,
    box: Tuple[int, int, int, int] | List[int],
    min_unet_sam_iou: float = 0.65,
    min_fill_ratio: float = 0.10,
    max_fill_ratio: float = 0.88,
    min_area_ratio: float = 0.35,
    max_area_ratio: float = 1.30,
) -> MaskVerificationResult:
    """Evaluates Condition C1: Deterministic GT-free Mask Verification.

    Computes observable agreement proxies between U-Net prediction and SAM mask:
      - unet_sam_iou = IoU(M_unet, M_sam)
      - prompt_box_fill_ratio = Area(M_sam ∩ PromptBox) / Area(PromptBox) (strictly in [0, 1])
      - sam_bbox_fill_ratio = Area(M_sam) / Area(SAM_Tight_BBox) (strictly in [0, 1])
      - area_ratio = Area(M_sam) / max(1, Area(M_unet))

    Guarantees monotonicity: if proxies clear all cutoffs, the mask is locked
    immediately (Delta IoU = 0.0). If discordance or box-explosion is observed,
    refinement is triggered.
    """
    m_unet = (unet_mask > 0).astype(np.uint8)
    m_sam = (sam_mask > 0).astype(np.uint8)

    inter = int(np.logical_and(m_unet, m_sam).sum())
    union = int(np.logical_or(m_unet, m_sam).sum())
    unet_sam_iou = float(inter / union) if union > 0 else 0.0

    sam_area = int(m_sam.sum())
    unet_area = int(m_unet.sum())

    x1, y1, x2, y2 = [int(v) for v in box]
    box_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))

    # prompt_box_fill_ratio: intersection of SAM mask inside prompt box / box area (strictly <= 1.0)
    clip_x1, clip_x2 = max(0, min(480, x1)), max(0, min(480, x2 + 1))
    clip_y1, clip_y2 = max(0, min(480, y1)), max(0, min(480, y2 + 1))
    sam_in_prompt_box = int(m_sam[clip_y1:clip_y2, clip_x1:clip_x2].sum())
    prompt_box_fill_ratio = float(min(1.0, max(0.0, sam_in_prompt_box / box_area)))

    # sam_bbox_fill_ratio: Area(M_sam) / Area(Tight BBox of M_sam) (strictly in [0, 1])
    y_idx, x_idx = np.where(m_sam > 0)
    if len(y_idx) > 0:
        sam_bx1, sam_bx2 = int(x_idx.min()), int(x_idx.max())
        sam_by1, sam_by2 = int(y_idx.min()), int(y_idx.max())
        sam_box_area = max(1, (sam_bx2 - sam_bx1 + 1) * (sam_by2 - sam_by1 + 1))
        sam_bbox_fill_ratio = float(min(1.0, max(0.0, sam_area / sam_box_area)))
    else:
        sam_bbox_fill_ratio = 0.0

    box_fill = prompt_box_fill_ratio
    area_ratio = float(sam_area / max(1, unet_area))

    triggers: List[str] = []
    if unet_sam_iou < min_unet_sam_iou:
        triggers.append(f"U-Net/SAM mask agreement {unet_sam_iou:.4f} < {min_unet_sam_iou}")
    if unet_sam_iou < 0.75 and box_fill < min_fill_ratio:
        triggers.append(f"Box fill ratio {box_fill:.4f} < {min_fill_ratio} (box under-filled or hollow)")
    if unet_sam_iou < 0.75 and box_fill > max_fill_ratio:
        triggers.append(f"Box fill ratio {box_fill:.4f} > {max_fill_ratio} (box-exploded or saturated)")
    if area_ratio < min_area_ratio:
        triggers.append(f"SAM/UNet area ratio {area_ratio:.4f} < {min_area_ratio} (severe under-segmentation)")
    if area_ratio > max_area_ratio:
        triggers.append(f"SAM/UNet area ratio {area_ratio:.4f} > {max_area_ratio} (severe over-segmentation)")

    needs_refinement = len(triggers) > 0
    return MaskVerificationResult(
        unet_sam_iou=unet_sam_iou,
        box_fill_ratio=box_fill,
        prompt_box_fill_ratio=prompt_box_fill_ratio,
        sam_bbox_fill_ratio=sam_bbox_fill_ratio,
        area_ratio=area_ratio,
        triggered_refinement=needs_refinement,
        trigger_reasons=triggers,
        is_locked=not needs_refinement,
    )
