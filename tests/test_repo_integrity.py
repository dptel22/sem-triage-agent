import hashlib
import json
from pathlib import Path

from scripts.make_results_table import check_file_matches
from src.api.sam_decoder import build_box_prompt, run_self_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_sam_decoder_bit_exactness():
    """(a) Verifies SAM 3.1 decoder self-test vectors and prompt builders."""
    assert run_self_test(verbose=False) is True
    prompt = build_box_prompt(10, 20, 30, 40, 480, 480, convention="half_open")
    assert prompt == "<0f>1<|box;x1=10;y1=20;x2=29;y2=39;w=480;h=480|>"


def test_splits_manifest_disjointness_and_isolation():
    """(b) Verifies splits_manifest.json has 4591 unique samples, zero overlap, and sharpened test isolation."""
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
    assert len(set(all_ids)) == 4591, f"Duplicates found in splits: {len(all_ids) - len(set(all_ids))}"

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
        assert len(containing_splits) == 1, f"Sample {sid} found in {len(containing_splits)} splits: {containing_splits}"
        assert containing_splits[0] in test_splits, f"Sample {sid} in non-test split: {containing_splits[0]}"


def test_readme_matches_script_output():
    """(c) Verifies README.md results tables match scripts/make_results_table.py output bit-for-bit."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "Missing README.md"
    assert check_file_matches(readme_path) is True, "README.md tables do not match JSON outputs!"


def test_gate0_checksums_match_checksums_txt():
    """(d) Verifies SHA-256 checksums of results/gate0/*.json against docs/checksums.txt."""
    checksums_path = PROJECT_ROOT / "docs" / "checksums.txt"
    assert checksums_path.exists(), "Missing docs/checksums.txt"

    lines = checksums_path.read_text(encoding="utf-8").splitlines()
    checked = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        assert len(parts) == 2, f"Malformed line in checksums.txt: {line}"
        expected_hash, rel_path = parts
        file_path = PROJECT_ROOT / rel_path.strip()

        assert file_path.exists(), f"File {file_path} listed in checksums.txt does not exist"
        h = hashlib.sha256(file_path.read_bytes()).hexdigest().upper()
        assert h == expected_hash.upper(), f"Hash mismatch for {rel_path}: expected {expected_hash}, got {h}"
        checked += 1

    assert checked >= 6, f"Expected at least 6 checksums to be verified, checked {checked}"
