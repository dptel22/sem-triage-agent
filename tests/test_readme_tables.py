"""Regression test: verifies that README results tables match scripts/make_results_table.py."""

from pathlib import Path
from scripts.make_results_table import check_file_matches

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_results_tables_match():
    """Verifies that all results tables in README.md match results/*.json bit-for-bit."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "Missing README.md"
    assert check_file_matches(readme_path) is True, "README.md tables do not match JSON outputs!"
