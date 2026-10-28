"""Tests for Carinthia-S dataset loader and preprocessing utilities."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.dataset.loader import (
    CarinthiaDataset,
    CarinthiaLoader,
    CarinthiaSDataset,
    compute_bounding_box,
    load_dataset,
    preprocess_mask,
)
import src.dataset.carinthia_s as carinthia_s

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "carinthia_s"


def test_unified_binarization_rule_and_border_zeroing():
    """Verifies mask >= 128 thresholding and zeroing of right-hand border (x >= 477)."""
    # Create synthetic 480x480 mask
    mask = np.zeros((480, 480), dtype=np.uint8)

    # Sub-threshold pixel (127) should become 0
    mask[50, 50] = 127
    # Threshold pixel (128) should become 1
    mask[50, 51] = 128
    # High pixel (255) in valid area should become 1
    mask[100, 200] = 255
    # High pixels (255) in border artifact columns (x >= 477) should become 0
    mask[100, 477] = 255
    mask[100, 478] = 255
    mask[100, 479] = 255

    bin_mask = preprocess_mask(mask, threshold=128, zero_border=True)

    assert bin_mask[50, 50] == 0, "Sub-threshold pixel (127) was not zeroed"
    assert bin_mask[50, 51] == 1, "Threshold pixel (128) was not set to 1"
    assert bin_mask[100, 200] == 1, "Foreground pixel was not set to 1"
    assert bin_mask[100, 477] == 0, "Border artifact pixel (x=477) was not zeroed"
    assert bin_mask[100, 478] == 0, "Border artifact pixel (x=478) was not zeroed"
    assert bin_mask[100, 479] == 0, "Border artifact pixel (x=479) was not zeroed"
    assert bin_mask[:, 477:].sum() == 0, "Border artifact region contains non-zero pixels"


def test_secondary_binarization():
    """Verifies secondary binarization (mask > 0) with zeroed border."""
    mask = np.zeros((480, 480), dtype=np.uint8)
    mask[10, 10] = 10
    mask[10, 478] = 200

    sec_mask = preprocess_mask(mask, threshold=0, zero_border=True)
    assert sec_mask[10, 10] == 1
    assert sec_mask[10, 478] == 0


def test_bounding_box_conventions():
    """Verifies inclusive vs half-open bounding box conventions."""
    mask = np.zeros((480, 480), dtype=np.uint8)
    # Defect from x in [50, 99], y in [100, 149] -> width 50, height 50
    mask[100:150, 50:100] = 200
    # Artifact on border
    mask[100:150, 478] = 255

    tight_inc, _ = compute_bounding_box(mask, pad_ratio=0.0, convention="inclusive")
    tight_half, _ = compute_bounding_box(mask, pad_ratio=0.0, convention="half_open")

    # Inclusive should be [50, 100, 99, 149]
    assert tight_inc == (50, 100, 99, 149)
    # Half-open should be [50, 100, 100, 150]
    assert tight_half == (50, 100, 100, 150)

    # Empty mask should return (None, None)
    empty_mask = np.zeros((480, 480), dtype=np.uint8)
    assert compute_bounding_box(empty_mask) == (None, None)


def test_carinthia_s_alias_module():
    """Verifies that src.dataset.carinthia_s exports match."""
    assert carinthia_s.CarinthiaSDataset is CarinthiaSDataset
    assert hasattr(carinthia_s, "preprocess_mask")
    assert hasattr(carinthia_s, "compute_bounding_box")


@pytest.mark.skipif(not (DATA_DIR / "carinthia-s.csv").exists(), reason="carinthia-s.csv not found")
def test_dataset_loader_with_csv():
    """Verifies loading real carinthia-s.csv with CarinthiaLoader and CarinthiaDataset."""
    assert CarinthiaDataset is CarinthiaLoader
    loader = load_dataset(DATA_DIR)
    assert isinstance(loader, CarinthiaLoader)
    assert len(loader) == 4591

    sample = loader[0]
    assert "image" in sample
    assert "mask" in sample
    assert "label" in sample
    assert isinstance(sample["image"], Image.Image)

    preprocessed = loader.get_preprocessed_mask(0)
    assert preprocessed.shape == (480, 480)
    assert preprocessed[:, 477:].sum() == 0
