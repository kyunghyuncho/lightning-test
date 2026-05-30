"""Hugging Face Hub authentication helpers."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv, set_key

CONFIG_DIR = Path.home() / ".config" / "object_detector"
ENV_FILE = CONFIG_DIR / ".env"


def init_hf_environment() -> None:
    """Load Hugging Face credentials from the isolated config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch(mode=0o600)
    load_dotenv(ENV_FILE, override=True)


def hf_token() -> str | None:
    """Return the configured Hugging Face token, if any."""
    init_hf_environment()
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def apply_hf_token_to_environ() -> str | None:
    """Ensure Hugging Face libraries can read the token from the environment."""
    token = hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def ensure_hf_credentials(*, interactive: bool = True) -> None:
    """Prompt for an optional Hugging Face token when one is not configured."""
    init_hf_environment()
    if hf_token():
        apply_hf_token_to_environ()
        return

    click.secho("\n=== HUGGING FACE ACCESS (OPTIONAL) ===", fg="cyan", bold=True)
    click.echo("1. Create a token at: https://huggingface.co/settings/tokens")
    click.echo("2. A token improves download rate limits and unlocks gated models.")
    click.echo("3. Public models such as facebook/detr-resnet-50 work without a token.\n")

    if not interactive:
        return

    if click.confirm("Configure Hugging Face token now?", default=False):
        token = click.prompt("Enter your HF_TOKEN", type=str, hide_input=True)
        set_key(str(ENV_FILE), "HF_TOKEN", token)
        load_dotenv(ENV_FILE, override=True)
        apply_hf_token_to_environ()
