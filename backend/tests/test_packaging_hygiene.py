from __future__ import annotations

from pathlib import Path


def test_gitignore_contains_transient_artifact_rules():
    root = Path(__file__).resolve().parents[2]
    content = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".pytest_cache/" in content
    assert "public/*.png" in content


def test_pack_script_excludes_transient_artifacts():
    root = Path(__file__).resolve().parents[2]
    content = (root / "pack.ps1").read_text(encoding="utf-8").lower()
    assert ".pytest_cache" in content
    assert "__pycache__" in content
    assert "public\\*.png" in content
