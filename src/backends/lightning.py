"""Lightning Studio Jobs remote inference backend."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import click

from src.backends.base import BaseInferenceBackend
from src.core.hf_auth import apply_hf_token_to_environ, hf_token
from src.core.images import SUPPORTED_IMAGE_EXTENSIONS, list_image_files, no_images_error_message
from src.core.lightning_config import (
    build_studio,
    ensure_lightning_studio_config,
    resolve_lightning_machine,
)

REMOTE_INPUT = "data/input_photos"
REMOTE_OUTPUT = "data/output"
REMOTE_SCRIPT = "data/run_detection.py"


class LightningJobsBackend(BaseInferenceBackend):
    """Provision a Lightning Studio on GPU and run detection there."""

    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        user_id = os.getenv("LIGHTNING_USER_ID")
        api_key = os.getenv("LIGHTNING_API_KEY")
        if not user_id or not api_key:
            raise RuntimeError(
                "LIGHTNING_USER_ID and LIGHTNING_API_KEY must be set. "
                "Run the CLI again to configure credentials interactively."
            )

        ensure_lightning_studio_config(interactive=False)
        machine = resolve_lightning_machine()

        input_path = Path(input_dir)
        image_files = list_image_files(input_path)
        if not image_files:
            raise FileNotFoundError(no_images_error_message(input_dir))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        studio = build_studio()
        click.echo(f"Starting Lightning Studio on {machine}...")
        studio.start(machine)

        try:
            click.echo(f"Studio machine: {studio.machine}")
            apply_hf_token_to_environ()
            if token := hf_token():
                studio.set_env({"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token})

            studio.run(f"mkdir -p {REMOTE_INPUT} {REMOTE_OUTPUT}")
            studio.upload_folder(str(input_path), remote_path=REMOTE_INPUT)
            self._upload_runner_script(studio, model_id)
            click.echo(f"Running object detection on Lightning Studio ({studio.machine})...")
            output, exit_code = studio.run_with_exit_code(f"python {REMOTE_SCRIPT}")
            if exit_code != 0:
                raise RuntimeError(
                    f"Lightning detection script failed with exit code {exit_code}. "
                    f"Remote output:\n{output}\n"
                    "Check the Studio logs in the Lightning console."
                )
            if output.strip():
                click.echo(output)
            studio.download_folder(REMOTE_OUTPUT, target_path=str(output_path))
        finally:
            studio.stop()

    def _upload_runner_script(self, studio, model_id: str) -> None:
        script = textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path

            import torch
            from PIL import Image
            from transformers import pipeline

            input_dir = Path({REMOTE_INPUT!r})
            output_dir = Path({REMOTE_OUTPUT!r})
            output_dir.mkdir(parents=True, exist_ok=True)

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this Lightning Studio machine.")

            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            pipe = pipeline(
                "object-detection",
                model={model_id!r},
                device=0,
                token=token,
            )

            extensions = {set(SUPPORTED_IMAGE_EXTENSIONS)!r}

            for img_file in sorted(input_dir.iterdir()):
                if not img_file.is_file() or img_file.suffix.lower() not in extensions:
                    continue
                image = Image.open(img_file).convert("RGB")
                predictions = pipe(image)
                out_file = output_dir / f"{{img_file.stem}}_preds.json"
                with out_file.open("w", encoding="utf-8") as handle:
                    json.dump(predictions, handle, indent=4)
            """
        ).strip()
        local_script = Path(".lightning_runner.py")
        local_script.write_text(script, encoding="utf-8")
        try:
            studio.upload_file(str(local_script), remote_path=REMOTE_SCRIPT)
        finally:
            local_script.unlink(missing_ok=True)

        studio.run("pip install click transformers torch pillow tqdm python-dotenv accelerate timm")
