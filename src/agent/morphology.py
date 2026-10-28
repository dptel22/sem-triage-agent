"""Defect morphology feature extraction for semiconductor SEM triage."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MorphologyFeatures:
    """Quantitative morphological descriptors of a segmented defect."""

    area: int
    box_area: int
    box_fill_ratio: float
    aspect_ratio: float
    log_area: float
    norm_centroid_x: float
    norm_centroid_y: float
    crop_mean_intensity: float
    crop_std_intensity: float
    contrast: float
    hist_8: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Flattens scalar descriptors into a 1D feature vector for retrieval."""
        scalars = [
            self.log_area,
            self.aspect_ratio,
            self.box_fill_ratio,
            self.norm_centroid_x,
            self.norm_centroid_y,
            self.crop_mean_intensity,
            self.crop_std_intensity,
            self.contrast,
        ]
        return np.array(scalars + self.hist_8, dtype=np.float32)


def compute_defect_morphology(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    box: Tuple[int, int, int, int] | List[int],
) -> MorphologyFeatures:
    """Computes quantitative morphology descriptors from an SEM image and defect mask.

    Args:
        image: Full 480x480 grayscale image (PIL or numpy array).
        mask: Full 480x480 binary mask (0 or 1, uint8 or bool).
        box: [x1, y1, x2, y2] bounding box coordinates.

    Returns:
        MorphologyFeatures dataclass.
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    else:
        img_arr = np.array(image, dtype=np.float32)
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

    mask_bin = (mask > 0).astype(np.uint8)
    area = int(mask_bin.sum())

    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    box_area = bw * bh
    box_fill = float(area / max(1, box_area))
    aspect_ratio = float(bw / max(1, bh))
    log_area = float(math.log(1.0 + area))

    # Centroid
    y_idx, x_idx = np.where(mask_bin > 0)
    if len(y_idx) > 0:
        cx = float(x_idx.mean()) / 480.0
        cy = float(y_idx.mean()) / 480.0
    else:
        cx = float((x1 + x2) / 2.0) / 480.0
        cy = float((y1 + y2) / 2.0) / 480.0

    # Crop analysis
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(480, x2 + 1), min(480, y2 + 1)
    crop_img = img_arr[cy1:cy2, cx1:cx2]
    crop_mask = mask_bin[cy1:cy2, cx1:cx2]

    if area > 0 and crop_mask.sum() > 0:
        defect_pixels = crop_img[crop_mask > 0]
        mean_int = float(defect_pixels.mean())
        std_int = float(defect_pixels.std())
        bg_pixels = crop_img[crop_mask == 0]
        bg_mean = float(bg_pixels.mean()) if len(bg_pixels) > 0 else 0.5
        contrast = float(abs(mean_int - bg_mean))
        # 8-bin histogram
        hist, _ = np.histogram(defect_pixels, bins=8, range=(0.0, 1.0), density=True)
        hist_norm = [float(h / (hist.sum() + 1e-6)) for h in hist]
    else:
        mean_int = float(crop_img.mean()) if crop_img.size > 0 else 0.5
        std_int = float(crop_img.std()) if crop_img.size > 0 else 0.0
        contrast = 0.0
        hist_norm = [0.125] * 8

    return MorphologyFeatures(
        area=area,
        box_area=box_area,
        box_fill_ratio=box_fill,
        aspect_ratio=aspect_ratio,
        log_area=log_area,
        norm_centroid_x=cx,
        norm_centroid_y=cy,
        crop_mean_intensity=mean_int,
        crop_std_intensity=std_int,
        contrast=contrast,
        hist_8=hist_norm,
    )
