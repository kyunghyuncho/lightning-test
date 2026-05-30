"""Lightning Studio configuration and teamspace resolution."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv, set_key

CONFIG_DIR = Path.home() / ".config" / "object_detector"
ENV_FILE = CONFIG_DIR / ".env"

DEFAULT_STUDIO_NAME = "inference-cluster-studio"
DEFAULT_LIGHTNING_MACHINE = "T4"


def resolve_lightning_machine():
    """Resolve the Lightning GPU machine type used for inference."""
    from lightning_sdk import Machine

    init_lightning_environment()
    machine_name = os.getenv("LIGHTNING_MACHINE", DEFAULT_LIGHTNING_MACHINE).upper()
    if hasattr(Machine, machine_name):
        return getattr(Machine, machine_name)

    supported = sorted(name for name in dir(Machine) if name.isupper() and not name.startswith("_"))
    raise ValueError(
        f"Unknown LIGHTNING_MACHINE '{machine_name}'. "
        f"Supported examples: {', '.join(supported[:8])}..."
    )


def init_lightning_environment() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch(mode=0o600)
    load_dotenv(ENV_FILE, override=True)


def _save_env_value(key: str, value: str) -> None:
    set_key(str(ENV_FILE), key, value)
    load_dotenv(ENV_FILE, override=True)


def _fetch_authed_username() -> str:
    from lightning_sdk.utils.resolve import _get_authed_user

    return _get_authed_user().name


def _fetch_accessible_teamspaces() -> list[str]:
    from lightning_sdk.utils.resolve import _get_teamspace_names_for_authed_user

    return _get_teamspace_names_for_authed_user()


def _prompt_teamspace_choice(teamspaces: list[str]) -> str:
    if len(teamspaces) == 1:
        return teamspaces[0]

    click.echo("\nAvailable Lightning teamspaces:")
    for name in teamspaces:
        click.echo(f"  • {name}")

    return click.prompt(
        "Select teamspace",
        type=click.Choice(teamspaces, case_sensitive=False),
    )


def ensure_lightning_studio_config(*, interactive: bool = True) -> None:
    """Ensure teamspace and owner settings required by the Lightning SDK exist."""
    init_lightning_environment()

    teamspace = os.getenv("LIGHTNING_TEAMSPACE")
    username = os.getenv("LIGHTNING_USERNAME")
    org = os.getenv("LIGHTNING_ORG")

    if teamspace and (username or org):
        return

    discovery_error: Exception | None = None
    try:
        if not username and not org:
            username = _fetch_authed_username()
            _save_env_value("LIGHTNING_USERNAME", username)

        if not teamspace:
            teamspaces = _fetch_accessible_teamspaces()
            if teamspaces:
                selected = (
                    _prompt_teamspace_choice(teamspaces)
                    if interactive and len(teamspaces) > 1
                    else teamspaces[0]
                )
                _save_env_value("LIGHTNING_TEAMSPACE", selected)
                teamspace = selected
    except Exception as exc:
        discovery_error = exc

    teamspace = os.getenv("LIGHTNING_TEAMSPACE")
    username = os.getenv("LIGHTNING_USERNAME")
    org = os.getenv("LIGHTNING_ORG")

    if teamspace and (username or org):
        return

    if not interactive:
        message = (
            "Lightning teamspace settings are incomplete. Set LIGHTNING_TEAMSPACE and "
            "LIGHTNING_USERNAME (or LIGHTNING_ORG), or run interactively."
        )
        if discovery_error is not None:
            raise RuntimeError(message) from discovery_error
        raise RuntimeError(message)

    click.secho("\n=== LIGHTNING TEAMSPACE CONFIGURATION REQUIRED ===", fg="cyan", bold=True)
    click.echo("Studios are created inside a Lightning teamspace owned by a user or org.")
    click.echo("Find these values in your Lightning console URL:")
    click.echo("  https://lightning.ai/<owner>/<teamspace>/...\n")

    if discovery_error is not None:
        click.secho(
            f"Could not auto-discover teamspaces: {discovery_error}",
            fg="yellow",
        )

    if not teamspace:
        teamspace = click.prompt("Enter your Lightning teamspace name", type=str)
        _save_env_value("LIGHTNING_TEAMSPACE", teamspace)

    if not username and not org:
        owner_type = click.prompt(
            "Teamspace owner type",
            type=click.Choice(["user", "organization"], case_sensitive=False),
            default="user",
        )
        if owner_type == "organization":
            org = click.prompt("Enter your Lightning organization name", type=str)
            _save_env_value("LIGHTNING_ORG", org)
        else:
            username = click.prompt("Enter your Lightning username", type=str)
            _save_env_value("LIGHTNING_USERNAME", username)


def build_studio():
    """Construct a Lightning Studio client with resolved teamspace context."""
    from lightning_sdk import Studio

    init_lightning_environment()

    teamspace = os.getenv("LIGHTNING_TEAMSPACE")
    username = os.getenv("LIGHTNING_USERNAME")
    org = os.getenv("LIGHTNING_ORG")
    studio_name = os.getenv("LIGHTNING_STUDIO_NAME", DEFAULT_STUDIO_NAME)

    if not teamspace:
        raise RuntimeError(
            "LIGHTNING_TEAMSPACE is not configured. Run the CLI interactively or set it in "
            f"{ENV_FILE}."
        )

    kwargs: dict[str, object] = {
        "name": studio_name,
        "teamspace": teamspace,
        "create_ok": True,
    }

    if org:
        kwargs["org"] = org
    elif username:
        kwargs["user"] = username
    else:
        kwargs["user"] = _fetch_authed_username()

    return Studio(**kwargs)
