"""Tests for the object detection CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from src.backends.local import LocalBackend
from src.main import init_environment, run_pipeline


@pytest.fixture
def sample_image_dir(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), color="red").save(image_dir / "sample.jpg")
    return image_dir


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "predictions"
    out.mkdir()
    return out


def test_local_backend_writes_predictions(sample_image_dir: Path, output_dir: Path) -> None:
    mock_predictions = [{"score": 0.95, "label": "person", "box": {"xmin": 1, "ymin": 2}}]

    with patch("src.backends.local.ObjectDetectorEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.process_image.return_value = mock_predictions
        mock_engine_cls.return_value = mock_engine

        LocalBackend().execute(
            input_dir=str(sample_image_dir),
            output_dir=str(output_dir),
            model_id="facebook/detr-resnet-50",
        )

    result_file = output_dir / "sample_preds.json"
    assert result_file.exists()
    assert json.loads(result_file.read_text(encoding="utf-8")) == mock_predictions


def test_local_backend_raises_when_no_images(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No JPEG images"):
        LocalBackend().execute(str(empty_dir), str(out_dir), "facebook/detr-resnet-50")


def test_cli_local_backend(sample_image_dir: Path, output_dir: Path) -> None:
    runner = CliRunner()
    mock_predictions = [{"score": 0.9, "label": "cat", "box": {"xmin": 0, "ymin": 0}}]

    with patch("src.backends.local.ObjectDetectorEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.process_image.return_value = mock_predictions
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            run_pipeline,
            [
                "--backend",
                "local",
                "--input-dir",
                str(sample_image_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert (output_dir / "sample_preds.json").exists()


def test_cli_interactive_mode(sample_image_dir: Path, output_dir: Path) -> None:
    runner = CliRunner()
    mock_predictions = [{"score": 0.9, "label": "cat", "box": {"xmin": 0, "ymin": 0}}]

    with patch("src.backends.local.ObjectDetectorEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.process_image.return_value = mock_predictions
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            run_pipeline,
            input="\n".join(
                [
                    "local",
                    str(sample_image_dir),
                    str(output_dir),
                    "1",
                    "y",
                ]
            )
            + "\n",
        )

    assert result.exit_code == 0, result.output
    assert (output_dir / "sample_preds.json").exists()


def test_cli_missing_required_flags_shows_usage() -> None:
    runner = CliRunner()
    result = runner.invoke(run_pipeline, ["--backend", "local", "--no-interactive"])
    assert result.exit_code != 0
    assert "Missing required options" in result.output


def test_init_environment_creates_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config" / "object_detector"
    env_file = config_dir / ".env"
    monkeypatch.setattr("src.main.CONFIG_DIR", config_dir)
    monkeypatch.setattr("src.main.ENV_FILE", env_file)

    init_environment()

    assert config_dir.is_dir()
    assert env_file.exists()
