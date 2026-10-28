"""Clean Dataset Loader for Carinthia-S SEM Defect Dataset.

Attribution: Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427,
CC BY 4.0, single production layer of unstructured wafers).

Enforces the Gate 1 unified binarization standard:
  1. Primary rule: mask >= 128 (secondary comparison: mask > 0)
  2. Right-hand border artifact zeroed out: x >= 477 on 480-width image (or w - 3)
Supports both inclusive [x1, y1, x2, y2] and half-open [x1, y1, x2 + 1, y2 + 1]
bounding box conventions.

Reuses and wraps existing verified logic from src.dataset.carinthia_s / carinthia_loader
to avoid any duplicate logic (Ponytail Decision Ladder: Codebase Reuse).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.dataset.carinthia_s import CarinthiaSDataset


def preprocess_mask(
    mask: Union[np.ndarray, Image.Image],
    threshold: int = 128,
    zero_border: bool = True,
) -> np.ndarray:
    """Preprocesses mask using unified Gate 1 criteria.

    1. Zeroes out the 3-pixel right-hand black border artifact (x >= 477 on 480-width image).
    2. Binarizes mask using unified criterion: mask >= threshold (default: 128).
       If threshold <= 0, applies secondary criterion (mask > 0).

    Args:
        mask: Input mask as PIL Image or numpy array.
        threshold: Binarization threshold (default: 128).
        zero_border: If True, zero out columns at x >= 477 (or w - 3). Default True.

    Returns:
        np.ndarray: uint8 binary mask (0 or 1) of shape (H, W).
    """
    return CarinthiaSDataset.preprocess_mask(mask, threshold=threshold, zero_border=zero_border)


def compute_bounding_box(
    mask_arr: Union[np.ndarray, Image.Image],
    pad_ratio: float = 0.10,
    convention: str = "inclusive",
    threshold: int = 128,
    zero_border: bool = True,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """Computes tight and padded bounding boxes from a mask array.

    Coordinates are 0-indexed: [x1, y1, x2, y2].

    Args:
        mask_arr: 2D numpy array (or 3D with 1 channel) or PIL Image.
        pad_ratio: Fractional padding around the bounding box (default 0.10 = 10%).
        convention: Bounding box coordinate convention:
            - "inclusive": returns [x1, y1, x2, y2] where x2 and y2 are inclusive endpoints.
            - "half_open": returns [x1, y1, x2 + 1, y2 + 1] suitable for PIL.Image.crop.
        threshold: Mask binarization threshold (default 128).
        zero_border: Whether to zero out x >= 477 artifact before computing box.

    Returns:
        Tuple of (tight_bbox, padded_bbox), or (None, None) if mask has no foreground pixels.
    """
    return CarinthiaSDataset.compute_bounding_box(
        mask_arr,
        pad_ratio=pad_ratio,
        convention=convention,
        threshold=threshold,
        zero_border=zero_border,
    )


class CarinthiaLoader(CarinthiaSDataset):
    """Clean dataset loader for Carinthia-S (`carinthia-s.csv`).

    Wraps CarinthiaSDataset with direct access to preprocessed masks,
    bounding box computation, and split filtering.
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/carinthia_s",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        super().__init__(root_dir=root_dir)
        self.transform = transform

    def get_preprocessed_mask(
        self,
        idx: int,
        threshold: int = 128,
        zero_border: bool = True,
    ) -> np.ndarray:
        """Returns the unified binarized mask for sample at index `idx`."""
        sample = self[idx]
        return preprocess_mask(sample["mask"], threshold=threshold, zero_border=zero_border)

    def get_bounding_box(
        self,
        idx: int,
        pad_ratio: float = 0.10,
        convention: str = "inclusive",
        threshold: int = 128,
        zero_border: bool = True,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """Returns (tight_bbox, padded_bbox) for sample at index `idx`."""
        sample = self[idx]
        return compute_bounding_box(
            sample["mask"],
            pad_ratio=pad_ratio,
            convention=convention,
            threshold=threshold,
            zero_border=zero_border,
        )

    def filter_by_split(self, split_ids: Union[List[str], set[str]]) -> List[Dict[str, Any]]:
        """Returns samples filtered by a collection of sample IDs."""
        id_set = set(split_ids)
        return [s for s in self.samples if s["filename"] in id_set]


# Aliases for convenience and backward compatibility
CarinthiaDataset = CarinthiaLoader


def load_dataset(
    root_dir: Union[str, Path] = "data/carinthia_s",
) -> CarinthiaLoader:
    """Convenience factory function to instantiate CarinthiaLoader."""
    return CarinthiaLoader(root_dir=root_dir)


__all__ = [
    "CarinthiaSDataset",
    "CarinthiaLoader",
    "CarinthiaDataset",
    "preprocess_mask",
    "compute_bounding_box",
    "load_dataset",
]
