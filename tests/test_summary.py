"""Tests for detection sample summaries."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from src.core.summary import (
    collect_prediction_reports,
    format_image_sample,
    print_detection_samples,
)


def test_format_image_sample_lists_top_detections() -> None:
    predictions = [
        {"label": "cat", "score": 0.91},
        {"label": "dog", "score": 0.42},
        {"label": "person", "score": 0.88},
    ]

    lines = format_image_sample("photo.png", predictions, min_score=0.5, limit=5)

    assert lines[0] == "photo.png"
    assert "  • cat (91.0%)" in lines
    assert "  • person (88.0%)" in lines
    assert not any("dog" in line for line in lines)


def test_format_image_sample_handles_empty_predictions() -> None:
    lines = format_image_sample("empty.jpg", [], min_score=0.5, limit=5)
    assert lines[-1] == "  (no objects above threshold)"


def test_collect_prediction_reports_reads_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "predictions"
    output_dir.mkdir()
    (output_dir / "cat_preds.json").write_text(
        json.dumps([{"label": "cat", "score": 0.9}]),
        encoding="utf-8",
    )

    reports = collect_prediction_reports(output_dir)

    assert reports == [("cat", [{"label": "cat", "score": 0.9}])]


def test_print_detection_samples_renders_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "predictions"
    output_dir.mkdir()
    (output_dir / "sample_preds.json").write_text(
        json.dumps([{"label": "person", "score": 0.95}]),
        encoding="utf-8",
    )

    runner = CliRunner()
    with runner.isolation():
        print_detection_samples(output_dir)

    # ClickRunner isolation captures stdout internally; function should not raise.
