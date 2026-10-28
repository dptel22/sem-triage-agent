import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_gate0_and_gate1_checksums():
    """Verifies SHA-256 checksums of all tracked JSON artifacts against docs/checksums.txt."""
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

    assert checked >= 10, f"Expected at least 10 checksums to be verified, checked {checked}"
