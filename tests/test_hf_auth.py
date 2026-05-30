"""Tests for Hugging Face authentication helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.hf_auth import apply_hf_token_to_environ, ensure_hf_credentials, hf_token


@pytest.fixture
def config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    config_dir = tmp_path / "config" / "object_detector"
    env_file = config_dir / ".env"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.core.hf_auth.CONFIG_DIR", config_dir)
    monkeypatch.setattr("src.core.hf_auth.ENV_FILE", env_file)
    return config_dir, env_file


def test_hf_token_reads_from_env_file(config_paths: tuple[Path, Path]) -> None:
    _, env_file = config_paths
    env_file.write_text("HF_TOKEN=hf_test_token\n", encoding="utf-8")

    assert hf_token() == "hf_test_token"


def test_apply_hf_token_to_environ(config_paths: tuple[Path, Path]) -> None:
    _, env_file = config_paths
    env_file.write_text("HF_TOKEN=hf_test_token\n", encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    try:
        apply_hf_token_to_environ()
        assert os.environ["HF_TOKEN"] == "hf_test_token"
        assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
    finally:
        monkeypatch.undo()


def test_ensure_hf_credentials_prompts_and_persists(
    config_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir, env_file = config_paths
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with (
        patch("src.core.hf_auth.click.confirm", return_value=True),
        patch("src.core.hf_auth.click.prompt", return_value="hf_saved_token"),
    ):
        ensure_hf_credentials(interactive=True)

    assert config_dir.is_dir()
    assert "hf_saved_token" in env_file.read_text(encoding="utf-8")
    assert hf_token() == "hf_saved_token"
