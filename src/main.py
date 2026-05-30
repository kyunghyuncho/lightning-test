"""Click CLI entrypoint and credential orchestration."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv, set_key

from src.completion import complete_directories
from src.core.hf_auth import ensure_hf_credentials
from src.core.lightning_config import ensure_lightning_studio_config
from src.core.summary import print_detection_samples
from src.prompts import prompt_directory

CONFIG_DIR = Path.home() / ".config" / "object_detector"
ENV_FILE = CONFIG_DIR / ".env"

BACKEND_CHOICES = ["local", "lightning", "modal"]
DEFAULT_MODEL_ID = "facebook/detr-resnet-50"
DEFAULT_OUTPUT_DIR = "./predictions"

MODEL_CHOICES = {
    "1": "facebook/detr-resnet-50",
    "2": "hustvl/yolos-tiny",
    "3": "custom",
}


def init_environment() -> None:
    """Ensure isolated credential storage exists outside the repository."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch(mode=0o600)
    load_dotenv(ENV_FILE)


def interactive_credential_check(backend: str) -> None:
    """Prompt for missing cloud credentials and persist them securely."""
    init_environment()

    if backend == "lightning":
        if not os.getenv("LIGHTNING_API_KEY") or not os.getenv("LIGHTNING_USER_ID"):
            click.secho("\n=== LIGHTNING STUDIO AUTHENTICATION REQUIRED ===", fg="cyan", bold=True)
            click.echo("1. Log in via your web console interface at: https://lightning.ai")
            click.echo("2. Navigate to Settings -> API Keys to generate active credentials.")
            click.echo("3. Alternatively, execute 'lightning login' from your terminal.\n")

            uid = click.prompt("Enter your cloud LIGHTNING_USER_ID", type=str)
            key = click.prompt("Enter your cloud LIGHTNING_API_KEY", type=str, hide_input=True)

            set_key(str(ENV_FILE), "LIGHTNING_USER_ID", uid)
            set_key(str(ENV_FILE), "LIGHTNING_API_KEY", key)
            load_dotenv(ENV_FILE, override=True)

        ensure_lightning_studio_config(interactive=True)

    elif backend == "modal":
        if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
            click.secho("\n=== MODAL LABS AUTHENTICATION REQUIRED ===", fg="cyan", bold=True)
            click.echo("1. Create an active development workspace profile via https://modal.com")
            click.echo("2. Initialize authentication locally by executing: 'uv run modal setup'\n")

            token_id = click.prompt("Enter your MODAL_TOKEN_ID", type=str)
            token_secret = click.prompt("Enter your MODAL_TOKEN_SECRET", type=str, hide_input=True)

            set_key(str(ENV_FILE), "MODAL_TOKEN_ID", token_id)
            set_key(str(ENV_FILE), "MODAL_TOKEN_SECRET", token_secret)
            load_dotenv(ENV_FILE, override=True)


def _prompt_backend() -> str:
    click.echo("\nAvailable compute backends:")
    click.echo("  local      — Run on this machine (CPU / MPS / CUDA)")
    click.echo("  lightning  — Lightning Studio batch job (cloud GPU)")
    click.echo("  modal      — Modal Labs serverless inference (cloud GPU)")
    return click.prompt(
        "\nSelect backend",
        type=click.Choice(BACKEND_CHOICES, case_sensitive=False),
        default="local",
        show_choices=True,
    )


def _prompt_input_dir() -> str:
    return prompt_directory("Path to input image folder", must_exist=True)


def _prompt_output_dir(default: str) -> str:
    return prompt_directory(
        "Path for output JSON predictions",
        must_exist=False,
        default=default,
    )


def _prompt_model_id(default: str) -> str:
    click.echo("\nSelect a detection model:")
    click.echo("  1) facebook/detr-resnet-50  (default, high accuracy)")
    click.echo("  2) hustvl/yolos-tiny         (faster, lighter)")
    click.echo("  3) Enter a custom Hugging Face model ID")

    choice = click.prompt("Model choice", type=click.Choice(["1", "2", "3"]), default="1")
    if choice == "3":
        return click.prompt("Hugging Face model ID", type=str, default=default)
    return MODEL_CHOICES[choice]


def _confirm_run(backend: str, input_dir: str, output_dir: str, model_id: str) -> bool:
    click.echo("\n--- Run configuration ---")
    click.echo(f"  Backend:    {backend}")
    click.echo(f"  Input dir:  {input_dir}")
    click.echo(f"  Output dir: {output_dir}")
    click.echo(f"  Model:      {model_id}")
    return click.confirm("\nProceed with object detection?", default=True)


def resolve_run_config(
    backend: str | None,
    input_dir: str | None,
    output_dir: str | None,
    model_id: str | None,
    *,
    interactive: bool,
) -> tuple[str, str, str, str]:
    """Fill in missing run parameters via prompts or defaults."""
    if interactive:
        click.secho("\nObject Detector CLI", fg="cyan", bold=True)
        click.echo("Configure your object detection pipeline.\n")

    resolved_backend = backend or (_prompt_backend() if interactive else None)
    resolved_input = input_dir or (_prompt_input_dir() if interactive else None)
    resolved_output = output_dir or (
        _prompt_output_dir(DEFAULT_OUTPUT_DIR) if interactive else DEFAULT_OUTPUT_DIR
    )
    resolved_model = model_id or (
        _prompt_model_id(DEFAULT_MODEL_ID) if interactive else DEFAULT_MODEL_ID
    )

    if resolved_backend is None or resolved_input is None:
        raise click.UsageError(
            "Missing required options. Pass --backend and --input-dir, "
            "or run without flags for interactive mode."
        )

    if interactive and not _confirm_run(
        resolved_backend, resolved_input, resolved_output, resolved_model
    ):
        raise click.Abort()

    return resolved_backend, resolved_input, resolved_output, resolved_model


@click.command()
@click.option(
    "--backend",
    type=click.Choice(BACKEND_CHOICES),
    default=None,
    help="Compute infrastructure target provider backend.",
)
@click.option(
    "--input-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    default=None,
    help="Path pointing to image folder directory.",
    shell_complete=complete_directories,
)
@click.option(
    "--output-dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    default=None,
    help="Local directory to write output JSON reports.",
    shell_complete=complete_directories,
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Hugging Face Model Hub identifier string.",
)
@click.option(
    "--interactive/--no-interactive",
    default=None,
    help="Force interactive prompts even when all options are supplied.",
)
@click.option(
    "--show-samples/--no-show-samples",
    default=True,
    help="Print detected objects per image after the run completes.",
)
@click.option(
    "--sample-min-score",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum confidence score to include in sample output.",
)
@click.option(
    "--sample-limit",
    type=int,
    default=5,
    show_default=True,
    help="Maximum detections to show per image in the sample output.",
)
def run_pipeline(
    backend: str | None,
    input_dir: str | None,
    output_dir: str | None,
    model_id: str | None,
    interactive: bool | None,
    show_samples: bool,
    sample_min_score: float,
    sample_limit: int,
) -> None:
    """Cross-backend object detection orchestration CLI."""
    use_interactive = (
        interactive if interactive is not None else (backend is None or input_dir is None)
    )

    backend, input_dir, output_dir, model_id = resolve_run_config(
        backend,
        input_dir,
        output_dir,
        model_id,
        interactive=use_interactive,
    )

    interactive_credential_check(backend)
    ensure_hf_credentials(interactive=use_interactive)

    if backend == "local":
        from src.backends.local import LocalBackend

        engine = LocalBackend()
    elif backend == "lightning":
        from src.backends.lightning import LightningJobsBackend

        engine = LightningJobsBackend()
    elif backend == "modal":
        from src.backends.modal_backend import ModalBackend

        engine = ModalBackend()
    else:
        raise click.ClickException(f"Unsupported backend: {backend}")

    click.echo(f"\nInitializing pipeline on backend: {backend}")
    engine.execute(input_dir=input_dir, output_dir=output_dir, model_id=model_id)

    if show_samples:
        print_detection_samples(
            output_dir,
            min_score=sample_min_score,
            limit=sample_limit,
        )

    click.secho("Workflow successfully processed and concluded.", fg="green", bold=True)


if __name__ == "__main__":
    run_pipeline()
