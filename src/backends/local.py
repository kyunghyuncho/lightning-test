"""Local CPU/MPS/CUDA inference backend."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from src.backends.base import BaseInferenceBackend
from src.core.detector import ObjectDetectorEngine
from src.core.images import list_image_files, no_images_error_message


class LocalBackend(BaseInferenceBackend):
    """Execute object detection on the local machine."""

    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list_image_files(input_path)
        if not image_files:
            raise FileNotFoundError(no_images_error_message(input_dir))

        engine = ObjectDetectorEngine(model_id=model_id)

        for img_file in tqdm(image_files, desc="Local inference"):
            results = engine.process_image(str(img_file))
            out_file = output_path / f"{img_file.stem}_preds.json"
            with out_file.open("w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=4)
