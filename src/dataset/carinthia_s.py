"""Carinthia-S dataset module.

Attribution: Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427,
CC BY 4.0, single production layer of unstructured wafers).
"""

from __future__ import annotations

from src.dataset.carinthia_loader import CarinthiaSDataset

preprocess_mask = CarinthiaSDataset.preprocess_mask
compute_bounding_box = CarinthiaSDataset.compute_bounding_box

__all__ = [
    "CarinthiaSDataset",
    "preprocess_mask",
    "compute_bounding_box",
]
