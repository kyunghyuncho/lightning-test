"""Tests for directory tab completion helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click

from src.completion import complete_directories
from src.main import run_pipeline


def test_complete_directories_lists_matching_folders(tmp_path: Path) -> None:
    (tmp_path / "photos").mkdir()
    (tmp_path / "papers").mkdir()
    (tmp_path / "notes.txt").touch()

    ctx = click.Context(run_pipeline)
    param = next(option for option in run_pipeline.params if option.name == "input_dir")

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        completions = complete_directories(ctx, param, "p")

    values = {item.value for item in completions}
    assert any("photos" in value for value in values)
    assert any("papers" in value for value in values)
    assert not any("notes.txt" in value for value in values)
