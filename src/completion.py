"""Shell and interactive tab completion helpers for directory paths."""

from __future__ import annotations

import os
from pathlib import Path

import click
from click.shell_completion import CompletionItem


def _resolve_search_dir(incomplete: str) -> tuple[Path, str]:
    expanded = os.path.expanduser(incomplete or "")
    path = Path(expanded)

    if incomplete.endswith(os.sep) or not path.name:
        search_dir = path if path.is_dir() else path.parent
        prefix = ""
    else:
        search_dir = path.parent
        prefix = path.name

    if search_dir in (Path("."), Path("")):
        search_dir = Path.cwd()
    elif not search_dir.is_absolute():
        search_dir = Path.cwd() / search_dir

    if not search_dir.is_dir():
        search_dir = Path.cwd()

    return search_dir, prefix


def complete_directories(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[click.shell_completion.CompletionItem]:
    """Suggest directories only for shell tab completion."""
    del ctx, param

    search_dir, prefix = _resolve_search_dir(incomplete)

    items: list[click.shell_completion.CompletionItem] = []
    try:
        entries = sorted(search_dir.iterdir(), key=lambda entry: entry.name.lower())
    except OSError:
        return items

    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith(prefix):
            continue

        if incomplete.endswith(os.sep) or not Path(incomplete or "").name:
            suggestion = f"{incomplete}{entry.name}{os.sep}"
        elif search_dir == Path.cwd() and prefix:
            suggestion = f"{entry.name}{os.sep}"
        else:
            suggestion = str(search_dir / entry.name) + os.sep

        items.append(CompletionItem(suggestion, type="dir"))

    return items
