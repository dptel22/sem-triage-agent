"""Create Zero-Leakage Dataset Split for Gate 1.

Reads data/carinthia_s/carinthia-s.csv (4,591 samples) and results/gate0/gate0_20_manifest.json (20 samples)
to produce a zero-leakage, reproducible, audited split manifest saved to results/gate1/splits_manifest.json.

Split specifications:
1. Rare Holdout (Classes 1, 2, 5: 67 total - Class 1: 55, Class 2: 8, Class 5: 4):
   - Kept completely separate from U-Net training and validation.
   - The 6 Gate 0 rare images are tagged with seen_in_gate0=True and kept in rare_holdout.
   - Stratified 5-fold cross-validation fold assignment (seed=42) and precomputed Leave-One-Out (LOO) indexing.
2. Class 6 (SEM tool misalignment / no visible defect: 227 total - 224 empty, 3 non-empty):
   - Separated from nominal IoU metrics.
   - class6_train: 181
   - class6_val: 23 (includes the 1 Gate 0 Class 6 sample, tagged seen_in_gate0=True)
   - class6_test: 23
3. Nominal Defect Classes (Classes 3 & 4: 4,297 total - Class 3: 4,008, Class 4: 289):
   - Stratified split (seed=42):
     nominal_train: 3,437 (80% - Class 3: 3,206, Class 4: 231)
     nominal_val: 430 (10% - Class 3: 401, Class 4: 29; includes all 13 Gate 0 nominal samples tagged seen_in_gate0=True)
     nominal_test: 430 (10% - Class 3: 401, Class 4: 29; zero Gate 0 samples)
4. Primary Ground-Truth Mask Rule:
   - Primary mask threshold is mask >= 128 with the 3px right border (x >= 477) zeroed out.
   - Secondary comparison rule is mask > 0 with x >= 477 zeroed out.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "carinthia_s"
CSV_PATH = DATA_DIR / "carinthia-s.csv"
GATE0_MANIFEST_PATH = PROJECT_ROOT / "results" / "gate0" / "gate0_20_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "gate1"
OUTPUT_SPLITS_PATH = OUTPUT_DIR / "splits_manifest.json"


def load_dataset_records(csv_path: Path) -> List[Dict[str, Any]]:
    """Loads all records from official carinthia-s.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            records.append({
                "id": row.get("filename", Path(row["image_path"]).stem),
                "class_label": int(row["label"]),
            })
    return records


def load_gate0_ids(gate0_path: Path) -> set[str]:
    """Loads sample IDs from Gate 0 manifest."""
    if not gate0_path.exists():
        raise FileNotFoundError(f"Missing {gate0_path}")

    with open(gate0_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"] for s in data["samples"]}


def create_gate1_splits(
    records: List[Dict[str, Any]],
    gate0_ids: set[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates zero-leakage splits with strict class separation and Gate 0 accounting."""
    records = sorted(records, key=lambda r: r["id"])

    # Split records by defect class
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        lbl = r["class_label"]
        by_class.setdefault(lbl, []).append(r)

    # 1. Rare Holdout (Classes 1, 2, 5)
    rare_records = []
    for lbl in [1, 2, 5]:
        rare_records.extend(by_class.get(lbl, []))
    rare_records = sorted(rare_records, key=lambda r: r["id"])

    # Stratified fold assignment (seed=42)
    rng = random.Random(seed)
    rare_holdout = []
    for lbl in [1, 2, 5]:
        cls_records = [r.copy() for r in rare_records if r["class_label"] == lbl]
        rng.shuffle(cls_records)
        for i, r in enumerate(cls_records):
            r["seen_in_gate0"] = r["id"] in gate0_ids
            r["fold"] = i % 5  # Stratified round-robin fold (0..4)
            rare_holdout.append(r)

    # Sort rare_holdout deterministically by id
    rare_holdout = sorted(rare_holdout, key=lambda r: r["id"])

    # 2. Class 6 (227 total: 1 in Gate 0)
    class6_all = [r.copy() for r in by_class.get(6, [])]
    class6_g0 = [r for r in class6_all if r["id"] in gate0_ids]
    class6_non_g0 = [r for r in class6_all if r["id"] not in gate0_ids]

    assert len(class6_g0) == 1, f"Expected 1 Class 6 in Gate 0, found {len(class6_g0)}"
    assert len(class6_non_g0) == 226, f"Expected 226 non-Gate 0 Class 6, found {len(class6_non_g0)}"

    rng = random.Random(seed)
    rng.shuffle(class6_non_g0)

    for r in class6_g0:
        r["seen_in_gate0"] = True
    for r in class6_non_g0:
        r["seen_in_gate0"] = False

    class6_val = class6_g0 + class6_non_g0[:22]
    class6_test = class6_non_g0[22:45]
    class6_train = class6_non_g0[45:]

    assert len(class6_val) == 23
    assert len(class6_test) == 23
    assert len(class6_train) == 181

    # 3. Nominal Classes (Classes 3 & 4)
    nominal_train = []
    nominal_val = []
    nominal_test = []

    for lbl, (n_train, n_val, n_test) in [(3, (3206, 401, 401)), (4, (231, 29, 29))]:
        cls_all = [r.copy() for r in by_class.get(lbl, [])]
        cls_g0 = [r for r in cls_all if r["id"] in gate0_ids]
        cls_non_g0 = [r for r in cls_all if r["id"] not in gate0_ids]

        expected_g0_count = 7 if lbl == 3 else 6
        assert len(cls_g0) == expected_g0_count, f"Class {lbl}: expected {expected_g0_count} in Gate 0, got {len(cls_g0)}"

        for r in cls_g0:
            r["seen_in_gate0"] = True
        for r in cls_non_g0:
            r["seen_in_gate0"] = False

        rng = random.Random(seed + lbl)
        rng.shuffle(cls_non_g0)

        needed_val_from_non_g0 = n_val - len(cls_g0)
        cls_val = cls_g0 + cls_non_g0[:needed_val_from_non_g0]
        cls_test = cls_non_g0[needed_val_from_non_g0:needed_val_from_non_g0 + n_test]
        cls_train = cls_non_g0[needed_val_from_non_g0 + n_test:]

        assert len(cls_val) == n_val, f"Class {lbl} val count mismatch: {len(cls_val)} != {n_val}"
        assert len(cls_test) == n_test, f"Class {lbl} test count mismatch: {len(cls_test)} != {n_test}"
        assert len(cls_train) == n_train, f"Class {lbl} train count mismatch: {len(cls_train)} != {n_train}"

        nominal_val.extend(cls_val)
        nominal_test.extend(cls_test)
        nominal_train.extend(cls_train)

    assert len(nominal_train) == 3437
    assert len(nominal_val) == 430
    assert len(nominal_test) == 430

    splits = {
        "nominal_train": sorted(nominal_train, key=lambda r: r["id"]),
        "nominal_val": sorted(nominal_val, key=lambda r: r["id"]),
        "nominal_test": sorted(nominal_test, key=lambda r: r["id"]),
        "class6_train": sorted(class6_train, key=lambda r: r["id"]),
        "class6_val": sorted(class6_val, key=lambda r: r["id"]),
        "class6_test": sorted(class6_test, key=lambda r: r["id"]),
        "rare_holdout": sorted(rare_holdout, key=lambda r: r["id"]),
    }

    return splits


def audit_splits(splits: Dict[str, List[Dict[str, Any]]], gate0_ids: set[str]) -> Dict[str, Any]:
    """Audits zero-leakage, split counts, and Gate 0 placement."""
    split_counts = {k: len(v) for k, v in splits.items()}
    total_samples = sum(split_counts.values())

    id_to_split: Dict[str, str] = {}
    duplicate_ids: List[str] = []
    for split_name, items in splits.items():
        for item in items:
            iid = item["id"]
            if iid in id_to_split:
                duplicate_ids.append(f"{iid} in both {id_to_split[iid]} and {split_name}")
            id_to_split[iid] = split_name

    g0_in_splits: Dict[str, List[str]] = {}
    for split_name, items in splits.items():
        g0_here = [item["id"] for item in items if item.get("seen_in_gate0", False)]
        g0_in_splits[split_name] = g0_here

    leakage_in_train_or_test = (
        len(g0_in_splits["nominal_train"])
        + len(g0_in_splits["class6_train"])
        + len(g0_in_splits["nominal_test"])
        + len(g0_in_splits["class6_test"])
    )

    class_breakdown: Dict[str, Dict[int, int]] = {}
    for split_name, items in splits.items():
        cb: Dict[int, int] = {}
        for item in items:
            lbl = item["class_label"]
            cb[lbl] = cb.get(lbl, 0) + 1
        class_breakdown[split_name] = dict(sorted(cb.items()))

    audit_result = {
        "total_samples": total_samples,
        "expected_total": 4591,
        "total_matches": total_samples == 4591,
        "duplicate_count": len(duplicate_ids),
        "duplicates": duplicate_ids,
        "zero_leakage_verified": len(duplicate_ids) == 0 and leakage_in_train_or_test == 0,
        "gate0_distribution": {k: len(v) for k, v in g0_in_splits.items()},
        "gate0_leakage_count": leakage_in_train_or_test,
        "split_counts": split_counts,
        "class_breakdown": class_breakdown,
    }
    return audit_result


def main():
    print("=" * 80)
    print("GATE 1: ZERO-LEAKAGE DATASET SPLIT & AUDIT")
    print("=" * 80)

    records = load_dataset_records(CSV_PATH)
    gate0_ids = load_gate0_ids(GATE0_MANIFEST_PATH)
    print(f"Loaded {len(records)} records from {CSV_PATH}")
    print(f"Loaded {len(gate0_ids)} Gate 0 IDs from {GATE0_MANIFEST_PATH}")

    splits = create_gate1_splits(records, gate0_ids, seed=42)
    audit = audit_splits(splits, gate0_ids)

    print("\n--- Split Counts & Class Breakdown ---")
    for s_name, count in audit["split_counts"].items():
        cb_str = ", ".join(f"Class {k}: {v}" for k, v in audit["class_breakdown"][s_name].items())
        g0_count = audit["gate0_distribution"][s_name]
        print(f"  {s_name:<16}: {count:4d} samples (Gate 0 seen: {g0_count:2d}) | {cb_str}")

    print(f"\nTotal samples across all splits: {audit['total_samples']} (Expected: 4591)")
    print(f"Duplicate IDs across splits:     {audit['duplicate_count']}")
    print(f"Gate 0 leakage in train/test:    {audit['gate0_leakage_count']}")
    print(f"Zero-Leakage Status:             {'PASSED [VERIFIED]' if audit['zero_leakage_verified'] else 'FAILED'}")

    assert audit["total_matches"], f"Total count mismatch: {audit['total_samples']} != 4591"
    assert audit["duplicate_count"] == 0, f"Found duplicates: {audit['duplicates']}"
    assert audit["gate0_leakage_count"] == 0, "Gate 0 samples found in train or test splits!"

    # Format concise entries for JSON (<1 MB total)
    concise_splits: Dict[str, List[Dict[str, Any]]] = {}
    for s_name, items in splits.items():
        concise_items = []
        for it in items:
            row: Dict[str, Any] = {
                "id": it["id"],
                "class_label": it["class_label"],
            }
            if it.get("seen_in_gate0", False):
                row["seen_in_gate0"] = True
            if "fold" in it:
                row["fold"] = it["fold"]
            concise_items.append(row)
        concise_splits[s_name] = concise_items

    # Save splits manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "metadata": {
            "dataset_attribution": (
                "Carinthia-S public SEM defect dataset (DOI: 10.5281/zenodo.16895427, "
                "CC BY 4.0, single production layer of unstructured wafers)"
            ),
            "primary_ground_truth_rule": "mask >= 128 with right border artifact (x >= 477) zeroed out",
            "secondary_ground_truth_rule": "mask > 0 with right border artifact (x >= 477) zeroed out",
            "seed": 42,
            "path_conventions": {
                "image_path_template": "data/carinthia_s/images/{id}.jpg",
                "mask_path_template": "data/carinthia_s/masks/{id}.png",
            },
            "rare_holdout_loo_index": {
                "all_rare_ids": [r["id"] for r in splits["rare_holdout"]],
                "description": (
                    "Leave-One-Out (LOO) candidate pool for sample i is all_rare_ids excluding sample i. "
                    "Same-class candidate pool is all IDs of same class excluding sample i. "
                    "Stratified 5-fold CV fold index (0..4) is precomputed on each sample."
                ),
            },
            "audit_summary": audit,
        },
        "splits": concise_splits,
    }

    with open(OUTPUT_SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    file_size_kb = OUTPUT_SPLITS_PATH.stat().st_size / 1024
    print(f"\nSaved split manifest to {OUTPUT_SPLITS_PATH} ({file_size_kb:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
