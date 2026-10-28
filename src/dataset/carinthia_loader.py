"""Carinthia-S Dataset Loader.

Attribution: Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427,
CC BY 4.0, single production layer of unstructured wafers).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import csv
import numpy as np
from PIL import Image


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
                    "filename": row.get("filename", img_path.stem),
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": int(row["label"]),
                })

    def _load_from_dirs(self) -> None:
        """Fallback discovery from images and masks directories."""
        image_files = sorted(self.images_dir.glob("*.jpg"))
        for img_file in image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                self.samples.append({
                    "filename": img_file.stem,
                    "image_path": img_file,
                    "mask_path": mask_file,
                    "label": -1,  # Unknown without CSV
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        mask = Image.open(sample["mask_path"]).convert("L")
        return {
            "index": idx,
            "filename": sample["filename"],
            "image": image,
            "mask": mask,
            "label": sample["label"],
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

    @staticmethod
    def preprocess_mask(
        mask: np.ndarray | Image.Image,
        threshold: int = 128,
        zero_border: bool = True,
    ) -> np.ndarray:
        """Preprocesses mask according to Gate 1 unified criteria.

        1. Zeroes out the 3-pixel right-hand black border artifact (x >= 477 on 480-width image).
        2. Binarizes mask using unified primary criterion (default: mask >= 128) across ALL classes.
           If threshold <= 0, applies secondary criterion (mask > 0).

        Args:
            mask: Input mask as PIL Image or numpy array (H, W) or (H, W, C).
            threshold: Binarization threshold. Default is 128 (primary Gate 1 standard).
                       If <= 0, any non-zero pixel is treated as foreground (secondary comparison).
            zero_border: If True, zero out columns at x >= 477 (or w - 3). Default True.

        Returns:
            np.ndarray: uint8 binary mask (0 or 1, or 0 and 255 if needed) of shape (H, W).
        """
        if isinstance(mask, Image.Image):
            mask_arr = np.array(mask.convert("L"))
        else:
            mask_arr = np.array(mask)
            if mask_arr.ndim > 2:
                mask_arr = mask_arr[..., 0]

        mask_clean = mask_arr.copy()
        h, w = mask_clean.shape[:2]

        # Zero out the 3-pixel right-hand black border artifact documented in carinthia-s_dataset.html
        if zero_border and w >= 3:
            border_start = 477 if w == 480 else (w - 3)
            mask_clean[:, border_start:] = 0

        # Unified binarization
        if threshold > 0:
            bin_mask = (mask_clean >= threshold).astype(np.uint8)
        else:
            bin_mask = (mask_clean > 0).astype(np.uint8)

        return bin_mask

    @staticmethod
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
        bin_mask = CarinthiaSDataset.preprocess_mask(
            mask_arr, threshold=threshold, zero_border=zero_border
        )

        y_indices, x_indices = np.where(bin_mask > 0)
        if len(y_indices) == 0:
            return None, None

        h_img, w_img = bin_mask.shape[:2]
        x1, x2 = int(x_indices.min()), int(x_indices.max())
        y1, y2 = int(y_indices.min()), int(y_indices.max())

        # 10% padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = int(round(pad_ratio * bbox_w))
        pad_y = int(round(pad_ratio * bbox_h))

        pad_x1 = max(0, x1 - pad_x)
        pad_y1 = max(0, y1 - pad_y)
        pad_x2 = min(w_img - 1, x2 + pad_x)
        pad_y2 = min(h_img - 1, y2 + pad_y)

        if convention == "half_open":
            tight_bbox = (x1, y1, x2 + 1, y2 + 1)
            padded_bbox = (pad_x1, pad_y1, pad_x2 + 1, pad_y2 + 1)
        elif convention == "inclusive":
            tight_bbox = (x1, y1, x2, y2)
            padded_bbox = (pad_x1, pad_y1, pad_x2, pad_y2)
        else:
            raise ValueError(f"Unknown convention: {convention}. Expected 'inclusive' or 'half_open'.")

        return tight_bbox, padded_bbox
