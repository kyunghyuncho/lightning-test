"""Format and display object detection sample summaries."""

from __future__ import annotations

import json
from pathlib import Path

import click


def collect_prediction_reports(output_dir: Path) -> list[tuple[str, list[dict]]]:
    """Load prediction JSON artifacts from an output directory."""
    reports: list[tuple[str, list[dict]]] = []
    for path in sorted(output_dir.glob("*_preds.json")):
        image_name = path.name.removesuffix("_preds.json")
        try:
            predictions = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(predictions, list):
                reports.append((image_name, predictions))
        except json.JSONDecodeError:
            click.secho(
                f"Warning: Failed to parse prediction file {path.name}. Skipping.",
                fg="yellow",
                err=True,
            )
    return reports


def format_image_sample(
    image_name: str,
    predictions: list[dict],
    *,
    min_score: float = 0.5,
    limit: int = 5,
) -> list[str]:
    """Format top detections for a single image."""
    ranked = sorted(predictions, key=lambda item: item.get("score", 0.0), reverse=True)
    lines = [image_name]

    shown = 0
    for prediction in ranked:
        score = float(prediction.get("score", 0.0))
        if score < min_score:
            continue
        label = str(prediction.get("label", "unknown"))
        lines.append(f"  • {label} ({score:.1%})")
        shown += 1
        if shown >= limit:
            break

    if shown == 0:
        lines.append("  (no objects above threshold)")

    return lines


def print_detection_samples(
    output_dir: str | Path,
    *,
    min_score: float = 0.5,
    limit: int = 5,
    max_images: int | None = None,
) -> None:
    """Print a readable sample of detections grouped by image."""
    reports = collect_prediction_reports(Path(output_dir))
    if not reports:
        return

    click.echo("\nDetection samples")
    click.echo("─" * 40)

    display_reports = reports if max_images is None else reports[:max_images]
    for image_name, predictions in display_reports:
        for index, line in enumerate(
            format_image_sample(
                image_name,
                predictions,
                min_score=min_score,
                limit=limit,
            )
        ):
            if index == 0:
                click.secho(line, fg="cyan", bold=True)
            else:
                click.echo(line)
        click.echo()

    hidden = len(reports) - len(display_reports)
    if hidden > 0:
        click.echo(f"... and {hidden} more image(s). See JSON files in {output_dir}")
