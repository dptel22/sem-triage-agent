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
    def compute_bounding_box(mask_arr: np.ndarray, pad_ratio: float = 0.10) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """Computes tight (x1, y1, x2, y2) and padded bounding box from binary mask array.

        Returns (None, None) if mask has no foreground pixels.
        Coordinates are 0-indexed: [x1, y1, x2, y2].
        """
        if mask_arr.ndim > 2:
            mask_arr = mask_arr[..., 0]

        # Foreground is any non-zero pixel
        y_indices, x_indices = np.where(mask_arr > 0)
        if len(y_indices) == 0:
            return None, None

        h_img, w_img = mask_arr.shape[:2]
        x1, x2 = int(x_indices.min()), int(x_indices.max())
        y1, y2 = int(y_indices.min()), int(y_indices.max())
        tight_bbox = (x1, y1, x2, y2)

        # 10% padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = int(round(pad_ratio * bbox_w))
        pad_y = int(round(pad_ratio * bbox_h))

        pad_x1 = max(0, x1 - pad_x)
        pad_y1 = max(0, y1 - pad_y)
        pad_x2 = min(w_img - 1, x2 + pad_x)
        pad_y2 = min(h_img - 1, y2 + pad_y)
        padded_bbox = (pad_x1, pad_y1, pad_x2, pad_y2)

        return tight_bbox, padded_bbox
