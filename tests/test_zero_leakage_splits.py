"""Verification of Zero-Leakage Split Contract for Gate 1 splits_manifest.json."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = PROJECT_ROOT / "results" / "gate1" / "splits_manifest.json"
GATE0_PATH = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"
CHECKSUMS_PATH = PROJECT_ROOT / "docs" / "checksums.txt"


def test_splits_manifest_checksum():
    """Verifies that results/gate1/splits_manifest.json matches frozen SHA-256."""
    assert SPLITS_PATH.exists()
    assert CHECKSUMS_PATH.exists()

    expected_hash = None
    for line in CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        if "results/gate1/splits_manifest.json" in line:
            expected_hash = line.split()[0].upper()
            break

    assert expected_hash is not None, "splits_manifest.json not found in checksums.txt"
    actual_hash = hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest().upper()
    assert actual_hash == expected_hash, f"Checksum mismatch: {actual_hash} != {expected_hash}"


def test_zero_leakage_contract():
    """Verifies the complete zero-leakage split contract."""
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    splits = data["splits"]

    with open(GATE0_PATH, "r", encoding="utf-8") as f:
        gate0_samples = json.load(f)["samples"]
    gate0_ids = {s["id"] for s in gate0_samples}
    assert len(gate0_ids) == 20

    # 1. Total samples & uniqueness
    all_ids = []
    for split_name, samples in splits.items():
        all_ids.extend([s["id"] for s in samples])

    assert len(all_ids) == 4591
    assert len(set(all_ids)) == 4591

    # 2. Class 6 partitioning (227 total)
    c6_splits = {"class6_train", "class6_val", "class6_test"}
    c6_counts = {sp: sum(1 for s in splits[sp] if s["class_label"] == 6) for sp in splits}
    assert c6_counts["class6_train"] == 181
    assert c6_counts["class6_val"] == 23
    assert c6_counts["class6_test"] == 23
    for sp in splits:
        if sp not in c6_splits:
            assert c6_counts[sp] == 0, f"Found Class 6 in non-class6 split: {sp}"

    # 3. Rare defect classes (1, 2, 5: 67 total)
    rare_splits = {"rare_dev", "rare_eval_test"}
    assert len(splits["rare_dev"]) == 17
    assert len(splits["rare_eval_test"]) == 50
    assert len(splits["rare_dev"]) + len(splits["rare_eval_test"]) == 67

    for sp in rare_splits:
        for s in splits[sp]:
            assert s["class_label"] in {1, 2, 5}, f"Non-rare class {s['class_label']} in {sp}"

    for sp in splits:
        if sp not in rare_splits:
            for s in splits[sp]:
                assert s["class_label"] not in {1, 2, 5}, f"Rare defect found in non-rare split: {sp}"

    # 4. Nominal defect classes (3 & 4: 4,297 total)
    assert len(splits["nominal_train"]) == 3437
    assert len(splits["nominal_val"]) == 430
    assert len(splits["nominal_test"]) == 430

    # Stratification check for classes 3 & 4
    for sp, exp_c3, exp_c4 in [
        ("nominal_train", 3206, 231),
        ("nominal_val", 401, 29),
        ("nominal_test", 401, 29),
    ]:
        c3_count = sum(1 for s in splits[sp] if s["class_label"] == 3)
        c4_count = sum(1 for s in splits[sp] if s["class_label"] == 4)
        assert c3_count == exp_c3, f"{sp} Class 3 count mismatch: {c3_count} != {exp_c3}"
        assert c4_count == exp_c4, f"{sp} Class 4 count mismatch: {c4_count} != {exp_c4}"

    # 5. Gate 0 Accounting & Isolation
    gate0_found_by_split = {}
    for sp, samples in splits.items():
        found = [s for s in samples if s["id"] in gate0_ids]
        gate0_found_by_split[sp] = found

    assert len(gate0_found_by_split["nominal_val"]) == 13
    assert len(gate0_found_by_split["class6_val"]) == 1
    assert len(gate0_found_by_split["rare_dev"]) == 6

    # Verify zero in train and test splits
    for sp in ["nominal_train", "class6_train", "nominal_test", "class6_test", "rare_eval_test"]:
        assert len(gate0_found_by_split[sp]) == 0, f"Leakage: Gate 0 sample in {sp}!"

    # Verify all 20 have seen_in_gate0 == True
    all_g0_found = (
        gate0_found_by_split["nominal_val"]
        + gate0_found_by_split["class6_val"]
        + gate0_found_by_split["rare_dev"]
    )
    assert len(all_g0_found) == 20
    for s in all_g0_found:
        assert s.get("seen_in_gate0") is True
