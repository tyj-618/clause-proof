from __future__ import annotations

import tomllib
from pathlib import Path

import clauseproof


def test_package_importable() -> None:
    assert clauseproof.__version__


def test_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert clauseproof.__version__ == metadata["project"]["version"]
