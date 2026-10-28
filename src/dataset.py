"""Carinthia-S SEM Defect Dataset Loader and Binarization Protocol.

Attribution: Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427,
CC BY 4.0, single production layer of unstructured semiconductor wafers).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

# Exact dataset class definitions (Classes 1-5 active defects; Class 6 empty scan)
CLASS_NAMES: Dict[int, str] = {
    1: "ridges",
    2: "open",
    3: "nominal_3",
    4: "pits",
    5: "delamination",
    6: "no_defect_reimage",
}


def preprocess_mask(
    mask: np.ndarray | Image.Image,
    threshold: int = 128,
    zero_border: bool = True,
) -> np.ndarray:
    """Preprocesses ground truth / predicted mask according to Gate 1 unified criteria.

    1. Zeroes out the 3-pixel right-hand black border artifact (x >= 477 on 480-width image).
    2. Binarizes mask using unified primary criterion (default: mask >= 128) across ALL classes.
       If threshold <= 0, applies secondary comparison (mask > 0).

    Args:
        mask: Input mask as PIL Image or numpy array (H, W) or (H, W, C).
        threshold: Binarization threshold. Default 128 (primary Gate 1 standard).
                   If <= 0, any non-zero pixel is treated as foreground.
        zero_border: If True, zero out columns at x >= 477 (or w - 3). Default True.

    Returns:
        np.ndarray: uint8 binary mask ({0, 1}) of shape (H, W).
    """
    if isinstance(mask, Image.Image):
        mask_arr = np.array(mask.convert("L"))
    else:
        mask_arr = np.array(mask)
        if mask_arr.ndim > 2:
            mask_arr = mask_arr[..., 0]

    mask_clean = mask_arr.copy()
    h, w = mask_clean.shape[:2]

    # Zero out the 3-pixel right-hand black border artifact documented in Zenodo record
    if zero_border and w >= 3:
        border_start = 477 if w == 480 else (w - 3)
        mask_clean[:, border_start:] = 0

    # Unified binarization
    if threshold > 0:
        bin_mask = (mask_clean >= threshold).astype(np.uint8)
    else:
        bin_mask = (mask_clean > 0).astype(np.uint8)

    return bin_mask


def compute_bounding_box(
    mask_arr: np.ndarray,
    pad_ratio: float = 0.10,
    convention: str = "inclusive",
    threshold: int = 128,
    zero_border: bool = True,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """Computes tight and padded bounding boxes from a mask array.

    Coordinates are 0-indexed: [x1, y1, x2, y2].

    Args:
        mask_arr: 2D numpy array (or 3D with 1 channel).
        pad_ratio: Fractional padding around the bounding box (default 0.10 = 10%).
        convention: Bounding box coordinate convention:
            - "inclusive": returns [x1, y1, x2, y2] where x2 and y2 are the maximum
              foreground pixel indices included in the box (inclusive endpoints).
            - "half_open": returns [x1, y1, x2 + 1, y2 + 1] where x2 + 1 and y2 + 1
              are non-inclusive upper bounds, suitable directly for PIL.Image.crop(box).
        threshold: Mask binarization threshold (default 128). If <= 0, mask > 0.
        zero_border: Whether to zero out x >= 477 artifact before computing box.

    Returns:
        Tuple of (tight_bbox, padded_bbox), or (None, None) if mask has no foreground pixels.
    """
    bin_mask = preprocess_mask(mask_arr, threshold=threshold, zero_border=zero_border)
    h, w = bin_mask.shape[:2]

    rows = np.any(bin_mask > 0, axis=1)
    cols = np.any(bin_mask > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None, None

    y_min, y_max = int(np.where(rows)[0][[0, -1]][0]), int(np.where(rows)[0][[0, -1]][1])
    x_min, x_max = int(np.where(cols)[0][[0, -1]][0]), int(np.where(cols)[0][[0, -1]][1])

    pad_w = int(round((x_max - x_min + 1) * pad_ratio))
    pad_h = int(round((y_max - y_min + 1) * pad_ratio))

    max_valid_x = 476 if zero_border and w == 480 else (w - 1)
    px_min = max(0, x_min - pad_w)
    py_min = max(0, y_min - pad_h)
    px_max = min(max_valid_x, x_max + pad_w)
    py_max = min(h - 1, y_max + pad_h)

    if convention == "half_open":
        tight = (x_min, y_min, x_max + 1, y_max + 1)
        padded = (px_min, py_min, px_max + 1, py_max + 1)
    elif convention == "inclusive":
        tight = (x_min, y_min, x_max, y_max)
        padded = (px_min, py_min, px_max, py_max)
    else:
        raise ValueError(f"Unknown convention: {convention}. Use 'inclusive' or 'half_open'.")

    return tight, padded


class CarinthiaSDataset:
    """Dataset loader and inspector for Carinthia-S SEM defect dataset."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.csv_path = self.root_dir / "carinthia-s.csv"
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"
        self.samples: List[Dict[str, Any]] = []

        if self.csv_path.exists():
            self._load_from_csv()
        elif self.images_dir.exists() and self.masks_dir.exists():
            self._load_from_dirs()

    def _load_from_csv(self) -> None:
        """Load metadata directly from official carinthia-s.csv (semicolon delimited)."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = self.root_dir / row["image_path"]
                mask_path = self.root_dir / row["mask_path"]
                self.samples.append({
                    "id": row.get("filename", img_path.stem),
                    "filename": row.get("filename", img_path.stem),
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": int(row["label"]),
                    "class_name": CLASS_NAMES.get(int(row["label"]), f"class_{row['label']}"),
                })

    def _load_from_dirs(self) -> None:
        """Fallback discovery from images and masks directories."""
        image_files = sorted(self.images_dir.glob("*.jpg"))
        for img_file in image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                self.samples.append({
                    "id": img_file.stem,
                    "filename": img_file.stem,
                    "image_path": img_file,
                    "mask_path": mask_file,
                    "label": -1,
                    "class_name": "unknown",
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        mask = Image.open(sample["mask_path"]).convert("L")
        return {
            "index": idx,
            "id": sample["id"],
            "filename": sample["filename"],
            "image": image,
            "mask": mask,
            "label": sample["label"],
            "class_name": sample.get("class_name", ""),
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }

    def get_class_counts(self) -> Dict[int, int]:
        """Returns distribution of samples across defect classes."""
        counts: Dict[int, int] = {}
        for s in self.samples:
            lbl = s["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return dict(sorted(counts.items()))

    # Re-expose static helpers for backwards compatibility
    preprocess_mask = staticmethod(preprocess_mask)
    compute_bounding_box = staticmethod(compute_bounding_box)


if _TORCH_AVAILABLE:
    class SEMDataset(Dataset):
        """PyTorch Dataset for U-Net training and evaluation."""

        def __init__(self, sample_dicts: List[Dict[str, Any]], data_dir: str | Path):
            self.samples = sample_dicts
            self.data_dir = Path(data_dir)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
            item = self.samples[idx]
            sid = item["id"]
            img_p = self.data_dir / "images" / f"{sid}.jpg"
            mask_p = self.data_dir / "masks" / f"{sid}.png"

            img = Image.open(img_p).convert("L")
            img_arr = np.array(img, dtype=np.float32) / 255.0

            mask = Image.open(mask_p).convert("L")
            mask_bin = preprocess_mask(mask, threshold=128, zero_border=True)

            x = torch.from_numpy(img_arr).unsqueeze(0)  # (1, 480, 480)
            y = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)  # (1, 480, 480)
            return x, y, item
