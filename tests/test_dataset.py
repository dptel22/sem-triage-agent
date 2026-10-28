"""Unit tests for Carinthia-S dataset loader, binarization, and split integrity."""

import json
from pathlib import Path
import numpy as np

from src.dataset import (
    compute_bounding_box,
    preprocess_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_unified_binarization_rule_and_border_zeroing():
    """Verifies that mask >= 128 thresholding and x >= 477 right border zeroing are enforced."""
    raw = np.full((480, 480), 64, dtype=np.uint8)
    raw[100:200, 100:200] = 128
    raw[150:180, 475:480] = 255  # Crossing into right 3 border pixels

    cleaned = preprocess_mask(raw, threshold=128, zero_border=True)
    assert cleaned[100:200, 100:200].sum() == 100 * 100
    assert cleaned[:, 477:].sum() == 0  # 3 border columns zeroed
    assert cleaned[0:50, 0:50].sum() == 0  # < 128 zeroed


def test_bounding_box_conventions():
    """Verifies tight and padded bounding boxes with inclusive vs half-open conventions."""
    mask = np.zeros((480, 480), dtype=np.uint8)
    mask[50:100, 60:120] = 200

    tight_inc, padded_inc = compute_bounding_box(mask, pad_ratio=0.10, convention="inclusive")
    assert tight_inc == (60, 50, 119, 99)

    tight_half, padded_half = compute_bounding_box(mask, pad_ratio=0.10, convention="half_open")
    assert tight_half == (60, 50, 120, 100)


def test_splits_manifest_disjointness_and_isolation():
    """Verifies 8-split zero-leakage contract across 4,591 samples."""
    splits_path = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
    sharpened_path = PROJECT_ROOT / "results" / "gate1" / "sharpened_test_10_manifest.json"
    gate0_path = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"

    assert splits_path.exists(), "Missing splits_manifest.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    # 1. Total samples across all splits
    all_ids = []
    for split_name, samples in splits.items():
        all_ids.extend([s["id"] for s in samples])

    assert len(all_ids) == 4591, f"Expected 4591 samples, found {len(all_ids)}"
    assert len(set(all_ids)) == 4591, "Duplicate samples found across splits!"

    # 2. Zero overlap across all split pairs
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            set_i = set(s["id"] for s in splits[split_names[i]])
            set_j = set(s["id"] for s in splits[split_names[j]])
            overlap = set_i.intersection(set_j)
            assert len(overlap) == 0, f"Overlap between {split_names[i]} and {split_names[j]}: {overlap}"

    # 3. Sharpened 10-test sample isolation
    with open(sharpened_path, "r", encoding="utf-8") as f:
        sh_samples = json.load(f)["samples"]
    with open(gate0_path, "r", encoding="utf-8") as f:
        gate0_ids = set(s["id"] for s in json.load(f)["samples"])

    test_splits = {"rare_eval_test", "class6_test", "nominal_test"}
    assert len(sh_samples) == 10

    for s in sh_samples:
        sid = s["id"]
        assert sid not in gate0_ids, f"Sharpened test sample {sid} was seen in Gate 0!"
        containing_splits = [sp for sp, items in splits.items() if any(x["id"] == sid for x in items)]
        assert len(containing_splits) == 1, f"Sample {sid} found in {len(containing_splits)} splits"
        assert containing_splits[0] in test_splits, f"Sample {sid} in non-test split: {containing_splits[0]}"
