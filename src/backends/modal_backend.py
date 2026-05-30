"""Modal Labs serverless GPU inference backend."""

from __future__ import annotations

import io
import json
from pathlib import Path

import modal
from tqdm import tqdm

from src.backends.base import BaseInferenceBackend
from src.backends.local import IMAGE_GLOB

cuda_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "pillow>=10.3.0",
    "accelerate>=0.27.0",
)

app = modal.App("serverless-object-detection")


@app.cls(image=cuda_image, gpu="T4")
class ModalModelServer:
    """Serverless object-detection worker running on Modal GPU containers."""

    model_id: str = modal.parameter(default="facebook/detr-resnet-50")

    @modal.enter()
    def load_model(self) -> None:
        from transformers import pipeline

        self.pipe = pipeline("object-detection", model=self.model_id, device=0)

    @modal.method()
    def predict(self, image_bytes: bytes) -> list[dict]:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        predictions: list[dict] = self.pipe(image)
        return predictions


class ModalBackend(BaseInferenceBackend):
    """Execute object detection via Modal serverless GPU functions."""

    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(input_path.glob(IMAGE_GLOB))
        if not image_files:
            raise FileNotFoundError(f"No JPEG images found in {input_dir}")

        with app.run():
            server = ModalModelServer(model_id=model_id)
            for img_file in tqdm(image_files, desc="Modal inference"):
                img_bytes = img_file.read_bytes()
                predictions = server.predict.remote(img_bytes)

                out_file = output_path / f"{img_file.stem}_preds.json"
                with out_file.open("w", encoding="utf-8") as handle:
                    json.dump(predictions, handle, indent=4)
