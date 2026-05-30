"""Tab-completable directory prompts for interactive CLI mode."""

from __future__ import annotations

import sys
from pathlib import Path

import click


def _prompt_directory_click(
    message: str,
    *,
    must_exist: bool,
    default: str,
) -> str:
    while True:
        value = click.prompt(message, type=str, default=default or None)
        if not value and default:
            value = default
        path = Path(value).expanduser()
        if must_exist and not path.is_dir():
            click.secho(f"Directory not found: {value}", fg="red")
            continue
        return str(path.resolve() if path.exists() else path)


def _prompt_directory_prompt_toolkit(
    message: str,
    *,
    must_exist: bool,
    default: str,
) -> str:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter

    completer = PathCompleter(only_directories=True, expanduser=True)

    while True:
        value = prompt(
            f"{message}: ",
            completer=completer,
            complete_while_typing=True,
            default=default,
        ).strip()
        if not value and default:
            value = default
        path = Path(value).expanduser()
        if must_exist and not path.is_dir():
            click.secho(f"Directory not found: {value}", fg="red")
            continue
        return str(path.resolve() if path.exists() else path)


def prompt_directory(
    message: str,
    *,
    must_exist: bool = True,
    default: str = "",
) -> str:
    """Prompt for a directory path with tab completion when attached to a TTY."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _prompt_directory_prompt_toolkit(
                message,
                must_exist=must_exist,
                default=default,
            )
        except ImportError:
            pass

    return _prompt_directory_click(message, must_exist=must_exist, default=default)
