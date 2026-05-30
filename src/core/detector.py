"""Hugging Face object-detection pipeline wrapper."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import pipeline

from src.core.hf_auth import apply_hf_token_to_environ, hf_token


class ObjectDetectorEngine:
    """Unified inference engine for DETR / YOLOS-style object detection models."""

    def __init__(self, model_id: str = "facebook/detr-resnet-50") -> None:
        apply_hf_token_to_environ()
        token = hf_token()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.pipe = pipeline(
            "object-detection",
            model=model_id,
            device=self.device,
            token=token,
        )

    def process_image(self, image_path: str) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        predictions: list[dict] = self.pipe(image)
        return predictions
