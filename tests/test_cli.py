"""Tests for the object detection CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner
from PIL import Image

from src.backends.local import LocalBackend
from src.main import init_environment, run_pipeline
from src.prompts import prompt_directory


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

    with pytest.raises(FileNotFoundError, match="No supported image files"):
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
                    "n",
                ]
            )
            + "\n",
        )

    assert result.exit_code == 0, result.output
    assert (output_dir / "sample_preds.json").exists()


def test_local_backend_processes_png(sample_image_dir: Path, output_dir: Path) -> None:
    Image.new("RGBA", (64, 64), color="blue").save(sample_image_dir / "photo.png")
    mock_predictions = [{"score": 0.88, "label": "object", "box": {"xmin": 0, "ymin": 0}}]

    with patch("src.backends.local.ObjectDetectorEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.process_image.return_value = mock_predictions
        mock_engine_cls.return_value = mock_engine

        LocalBackend().execute(
            input_dir=str(sample_image_dir),
            output_dir=str(output_dir),
            model_id="facebook/detr-resnet-50",
        )

    assert (output_dir / "sample_preds.json").exists()
    assert (output_dir / "photo_preds.json").exists()
    assert mock_engine.process_image.call_count == 2


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


def test_prompt_directory_click_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    monkeypatch.setattr("src.prompts.sys.stdin.isatty", lambda: False)

    with patch("click.prompt", return_value=str(input_dir)) as mock_prompt:
        result = prompt_directory("Path to input image folder", must_exist=True)

    assert result == str(input_dir.resolve())
    mock_prompt.assert_called_once()


def test_input_dir_shell_completion_includes_directories(tmp_path: Path) -> None:
    child = tmp_path / "photos"
    child.mkdir()
    (tmp_path / "notes.txt").touch()

    ctx = click.Context(run_pipeline)
    input_dir_param = next(param for param in run_pipeline.params if param.name == "input_dir")

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        completions = list(input_dir_param.shell_complete(ctx, "p"))

    completion_values = {value.value for value in completions}
    assert any("photos" in value for value in completion_values)
    assert not any("notes.txt" in value for value in completion_values)
