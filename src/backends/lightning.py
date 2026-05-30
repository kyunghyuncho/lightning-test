"""Lightning Studio Jobs remote inference backend."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from src.backends.base import BaseInferenceBackend
from src.backends.local import IMAGE_GLOB

REMOTE_INPUT = "data/input_photos"
REMOTE_OUTPUT = "data/output"
REMOTE_SCRIPT = "data/run_detection.py"


class LightningJobsBackend(BaseInferenceBackend):
    """Provision a Lightning Studio, run a batch job, and sync results locally."""

    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        from lightning_sdk import Studio

        user_id = os.getenv("LIGHTNING_USER_ID")
        api_key = os.getenv("LIGHTNING_API_KEY")
        if not user_id or not api_key:
            raise RuntimeError(
                "LIGHTNING_USER_ID and LIGHTNING_API_KEY must be set. "
                "Run the CLI again to configure credentials interactively."
            )

        input_path = Path(input_dir)
        image_files = sorted(input_path.glob(IMAGE_GLOB))
        if not image_files:
            raise FileNotFoundError(f"No JPEG images found in {input_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        studio = Studio(name="inference-cluster-studio", create_ok=True)
        studio.start()

        try:
            studio.run(f"mkdir -p {REMOTE_INPUT} {REMOTE_OUTPUT}")
            studio.upload_folder(str(input_path), remote_path=REMOTE_INPUT)
            self._upload_runner_script(studio, model_id)
            self._launch_job(studio)
            studio.download_folder(REMOTE_OUTPUT, target_path=str(output_path))
        finally:
            studio.stop()

    def _upload_runner_script(self, studio, model_id: str) -> None:
        script = textwrap.dedent(
            f"""
            import json
            from pathlib import Path

            import torch
            from PIL import Image
            from transformers import pipeline

            input_dir = Path({REMOTE_INPUT!r})
            output_dir = Path({REMOTE_OUTPUT!r})
            output_dir.mkdir(parents=True, exist_ok=True)

            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("object-detection", model={model_id!r}, device=device)

            for img_file in sorted(input_dir.glob("*.[jJ][pP][gG]")):
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

        studio.run("pip install click transformers torch pillow tqdm python-dotenv accelerate")

    def _launch_job(self, studio) -> None:
        from lightning_sdk import Machine

        studio.install_plugin("jobs")
        jobs = studio.installed_plugins["jobs"]
        job = jobs.run(
            command=f"python {REMOTE_SCRIPT}",
            name="batch-detection-job",
            machine=Machine.T4,
        )

        wait_for_status = getattr(job, "wait_for_status", None)
        if callable(wait_for_status):
            job.wait_for_status("SUCCEEDED")
            return

        wait = getattr(job, "wait", None)
        if callable(wait):
            wait()
            return

        raise RuntimeError("Unable to wait for Lightning job completion via SDK.")
