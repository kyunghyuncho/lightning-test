"""Modal Labs serverless GPU inference backend (local client orchestration)."""

from __future__ import annotations

import json
from pathlib import Path

import modal
from tqdm import tqdm

from src.backends.base import BaseInferenceBackend
from src.backends.modal_app import ModalModelServer, app
from src.core.images import list_image_files, no_images_error_message


class ModalBackend(BaseInferenceBackend):
    """Execute object detection via Modal serverless GPU functions."""

    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        from src.core.hf_auth import apply_hf_token_to_environ, hf_token

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list_image_files(input_path)
        if not image_files:
            raise FileNotFoundError(no_images_error_message(input_dir))

        apply_hf_token_to_environ()
        secrets = []
        if token := hf_token():
            secrets = [modal.Secret.from_dict({"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token})]

        with app.run():
            server_cls = ModalModelServer.with_options(secrets=secrets)
            server = server_cls(model_id=model_id)
            img_bytes_list = [img_file.read_bytes() for img_file in image_files]
            predictions_list = list(
                tqdm(
                    server.predict.map(img_bytes_list),
                    total=len(img_bytes_list),
                    desc="Modal inference",
                )
            )

            for img_file, predictions in zip(image_files, predictions_list):
                out_file = output_path / f"{img_file.stem}_preds.json"
                with out_file.open("w", encoding="utf-8") as handle:
                    json.dump(predictions, handle, indent=4)
