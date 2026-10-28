import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_splits_manifest_integrity():
    """Verifies that all 4,591 samples are in exactly one split with zero duplicates and zero overlap."""
    splits_path = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
    assert splits_path.exists(), "Missing splits_manifest.json"

    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data["splits"]
    all_ids = []
    for split_name, samples in splits.items():
        s_ids = [s["id"] for s in samples]
        all_ids.extend(s_ids)

    # 1. Total count
    assert len(all_ids) == 4591, f"Expected 4591 samples, found {len(all_ids)}"

    # 2. Unique count (zero duplicates)
    unique_ids = set(all_ids)
    assert len(unique_ids) == 4591, f"Found {len(all_ids) - len(unique_ids)} duplicate IDs"

    # 3. Disjoint pair check (zero overlap)
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            set_i = set(s["id"] for s in splits[split_names[i]])
            set_j = set(s["id"] for s in splits[split_names[j]])
            overlap = set_i.intersection(set_j)
            assert len(overlap) == 0, f"Overlap between {split_names[i]} and {split_names[j]}: {overlap}"


def test_sharpened_test_sample_isolation():
    """Verifies all 10 sharpened test samples are in test splits and zero in Gate 0 manifest."""
    splits_path = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
    sharpened_path = PROJECT_ROOT / "results" / "gate1" / "sharpened_test_10_manifest.json"
    gate0_path = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]
    with open(sharpened_path, "r", encoding="utf-8") as f:
        sh_samples = json.load(f)["samples"]
    with open(gate0_path, "r", encoding="utf-8") as f:
        gate0_ids = set(s["id"] for s in json.load(f)["samples"])

    test_splits = {"rare_eval_test", "class6_test", "nominal_test"}

    assert len(sh_samples) == 10, f"Expected 10 test samples, found {len(sh_samples)}"

    for s in sh_samples:
        sid = s["id"]
        # Must not be in Gate 0
        assert sid not in gate0_ids, f"Sharpened test sample {sid} was seen in Gate 0!"

        # Must be in exactly one split and that split must be a test split
        containing_splits = [sp for sp, items in splits.items() if any(x["id"] == sid for x in items)]
        assert len(containing_splits) == 1, f"Sample {sid} found in {len(containing_splits)} splits: {containing_splits}"
        assert containing_splits[0] in test_splits, f"Sample {sid} in non-test split: {containing_splits[0]}"
