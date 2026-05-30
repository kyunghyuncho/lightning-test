"""Tests for supported image discovery."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.core.images import list_image_files, no_images_error_message


def test_list_image_files_supports_multiple_formats(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), color="red").save(tmp_path / "photo.jpg")
    Image.new("RGB", (8, 8), color="blue").save(tmp_path / "graphic.png")
    Image.new("RGB", (8, 8), color="green").save(tmp_path / "scan.tiff")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    image_files = list_image_files(tmp_path)

    assert [path.name for path in image_files] == ["graphic.png", "photo.jpg", "scan.tiff"]


def test_no_images_error_message_lists_supported_extensions(tmp_path: Path) -> None:
    message = no_images_error_message(tmp_path)

    assert "No supported image files found" in message
    assert ".png" in message
    assert ".jpg" in message
