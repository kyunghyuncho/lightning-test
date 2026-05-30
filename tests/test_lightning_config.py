"""Tests for Lightning teamspace configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.lightning_config import (
    build_studio,
    ensure_lightning_studio_config,
    resolve_lightning_machine,
)


@pytest.fixture
def config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / "config" / "object_detector"
    env_file = config_dir / ".env"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.core.lightning_config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("src.core.lightning_config.ENV_FILE", env_file)
    return env_file


def test_ensure_lightning_studio_config_discovers_teamspace(
    config_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = config_paths
    env_file.write_text(
        "LIGHTNING_USER_ID=test-user\nLIGHTNING_API_KEY=test-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LIGHTNING_TEAMSPACE", raising=False)
    monkeypatch.delenv("LIGHTNING_USERNAME", raising=False)
    monkeypatch.delenv("LIGHTNING_ORG", raising=False)

    with (
        patch("src.core.lightning_config._fetch_authed_username", return_value="alice"),
        patch(
            "src.core.lightning_config._fetch_accessible_teamspaces",
            return_value=["research-lab"],
        ),
    ):
        ensure_lightning_studio_config(interactive=False)

    contents = env_file.read_text(encoding="utf-8")
    assert "LIGHTNING_TEAMSPACE" in contents and "research-lab" in contents
    assert "LIGHTNING_USERNAME" in contents and "alice" in contents


def test_build_studio_passes_teamspace_and_user(config_paths: Path) -> None:
    env_file = config_paths
    env_file.write_text(
        "\n".join(
            [
                "LIGHTNING_TEAMSPACE=research-lab",
                "LIGHTNING_USERNAME=alice",
            ]
        ),
        encoding="utf-8",
    )

    mock_studio = MagicMock()
    with patch("lightning_sdk.Studio", return_value=mock_studio) as mock_studio_cls:
        studio = build_studio()

    assert studio is mock_studio
    mock_studio_cls.assert_called_once_with(
        name="inference-cluster-studio",
        teamspace="research-lab",
        create_ok=True,
        user="alice",
    )


def test_build_studio_requires_teamspace(
    config_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_paths.write_text("LIGHTNING_USERNAME=alice\n", encoding="utf-8")
    monkeypatch.delenv("LIGHTNING_TEAMSPACE", raising=False)

    with pytest.raises(RuntimeError, match="LIGHTNING_TEAMSPACE"):
        build_studio()


def test_resolve_lightning_machine_defaults_to_t4(
    config_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LIGHTNING_MACHINE", raising=False)

    from lightning_sdk import Machine

    assert resolve_lightning_machine() == Machine.T4


def test_resolve_lightning_machine_reads_env(config_paths: Path) -> None:
    config_paths.write_text("LIGHTNING_MACHINE=T4_X_2\n", encoding="utf-8")

    from lightning_sdk import Machine

    assert resolve_lightning_machine() == Machine.T4_X_2
