"""Modal app definition for remote GPU inference (minimal container imports)."""

import io

import modal

cuda_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "pillow>=10.3.0",
    "accelerate>=0.27.0",
    "timm>=0.9.0",
)

app = modal.App("serverless-object-detection")


@app.cls(image=cuda_image, gpu="T4")
class ModalModelServer:
    """Serverless object-detection worker running on Modal GPU containers."""

    model_id: str = modal.parameter(default="facebook/detr-resnet-50")

    @modal.enter()
    def load_model(self) -> None:
        import os

        from transformers import pipeline

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.pipe = pipeline(
            "object-detection",
            model=self.model_id,
            device=0,
            token=token,
        )

    @modal.method()
    def predict(self, image_bytes: bytes) -> list[dict]:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        predictions: list[dict] = self.pipe(image)
        return predictions
