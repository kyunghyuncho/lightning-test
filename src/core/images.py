"""Shared helpers for discovering supported input images."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
    }
)


def list_image_files(directory: Path) -> list[Path]:
    """Return supported image files in ``directory``, sorted by filename."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def no_images_error_message(directory: str | Path) -> str:
    extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
    return f"No supported image files found in {directory}. Supported extensions: {extensions}"
