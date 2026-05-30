# Implementation Plan: Multimodal Object Detection CLI (Local, Lightning Studio, Modal)

This document provides a comprehensive, production-grade architecture design and implementation plan for a Python CLI tool capable of executing object detection on images across three separate compute backends: **Local Machine**, **Lightning Studio Jobs**, and **Modal Labs Serverless Inference**.

---

## 1. System Architecture Overview

The system is designed as a monolithic Python CLI tool driven by `click`. Dependencies are managed strictly using `uv` to ensure deterministic environments.

```
                                    +--------------> [ Local Compute (CPU/MPS) ]
                                    |
[ CLI Engine ] === (Backend Router) +--------------> [ Lightning Studio Job (SDK API) ]
                                    |
                                    +--------------> [ Modal Labs Serverless (gRPC) ]

```

### Architectural Trade-offs Matrix

| Feature / Backend | Local Machine | Lightning Studio Jobs | Modal Labs |
| --- | --- | --- | --- |
| **Compute Semantics** | Persistent local loops | Asynchronous Batch Job | Ephemeral Serverless Function |
| **Hardware Target** | Mac OS (`mps` / `cpu`) | Cloud GPU (e.g., `T4`, `A10G`) | On-Demand GPU (e.g., `T4`, `A100`) |
| **Data Transport** | Direct Disk I/O | Programmatic SDK Sync | Ephemeral Network Payload / Vol |
| **Cold Start Latency** | $0$ seconds | $2$ to $5$ minutes (Provisioning) | $<2$ seconds |

---

## 2. Directory Structure

```
object_detector_cli/
│
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
│
├── src/
│   ├── __init__.py
│   ├── main.py                # Click CLI Entrypoint & Configuration Handlers
│   ├── core/
│   │   ├── __init__.py
│   │   └── detector.py        # Hugging Face pipeline wrapping Detr/YOLOS
│   │
│   └── backends/
│       ├── __init__.py
│       ├── base.py            # Abstract Base Class defining execute interface
│       ├── local.py           # Local computation routine
│       ├── lightning.py       # Lightning SDK wrapper executing remote jobs
│       └── modal_backend.py   # Modal app orchestration logic
│
└── tests/                     # Local test assertions

```

---

## 3. Configuration Management & Security

To prevent secrets exposure within a GitHub repository, all access keys, project tokens, and user IDs are managed exclusively via standard Unix Environment variables or a secure, non-committed `.env` file located in the user's home directory (`~/.config/object_detector/.env`) with restricted filesystem permissions.

### Safe Credential Matrix

* **Lightning AI**: Requires `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY`.
* **Modal Labs**: Requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
* **Hugging Face**: Requires `HF_TOKEN` (optional, needed for gated model variants).

---

## 4. Phase-by-Phase Implementation Blueprint

### Phase 1: Environment Definition & Dependencies (`pyproject.toml`)

Initialize the project environment utilizing the `uv` package manager targeting Python 3.11+.

```toml
[project]
name = "object-detector-cli"
version = "0.1.0"
description = "A production CLI for cross-backend object detection pipeline orchestration."
requires-python = ">=3.11"
dependencies = [
    "click>=8.1.7",
    "transformers>=4.40.0",
    "torch>=2.2.0",
    "pillow>=10.3.0",
    "lightning-sdk>=0.3.0",
    "modal>=0.62.0",
    "python-dotenv>=1.0.1",
    "tqdm>=4.66.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
]

```

### Phase 2: Core Vision Logic (`src/core/detector.py`)

Implement a unified inference class utilizing Hugging Face `transformers` with a state-of-the-art model suited for object detection tasks (e.g., `facebook/detr-resnet-50` or `huggingface/yolos-tiny`).

```python
import torch
from PIL import Image
from transformers import pipeline

class ObjectDetectorEngine:
    def __init__(self, model_id: str = "facebook/detr-resnet-50"):
        # Select Apple Silicon GPU acceleration if available, otherwise fallback to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            
        self.pipe = pipeline(
            "object-detection", 
            model=model_id, 
            device=0 if self.device == "cuda" else -1
        )

    def process_image(self, image_path: str) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        predictions = self.pipe(image)
        return predictions

```

### Phase 3: Abstract Execution Layer & Local Backend (`src/backends/`)

Every computational target must conform to a standardized abstract base class interface execution schema.

```python
# src/backends/base.py
from abc import ABC, abstractmethod

class BaseInferenceBackend(ABC):
    @abstractmethod
    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        pass

```

```python
# src/backends/local.py
import os
import json
from pathlib import Path
from src.backends.base import BaseInferenceBackend
from src.core.detector import ObjectDetectorEngine

class LocalBackend(BaseInferenceBackend):
    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        engine = ObjectDetectorEngine(model_id=model_id)
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for img_file in input_path.glob("*.[jJ][pP][gG]"):
            results = engine.process_image(str(img_file))
            out_file = output_path / f"{img_file.stem}_preds.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)

```

### Phase 4: Modal Labs Inference Implementation (`src/backends/modal_backend.py`)

Modal manages dynamic environment setups purely through programmatic API definitions. It requires wrapping inference into an active application object structure (`modal.App`).

```python
import modal
import os
from pathlib import Path

# Build container context with exact PyPI packages required
cuda_image = modal.Image.debian_slim().uv_pip_install(
    "transformers", "torch", "pillow", "accelerate"
)

app = modal.App("serverless-object-detection")

@app.cls(image=cuda_image, gpu="T4")
class ModalModelServer:
    def __enter__(self):
        from transformers import pipeline
        self.pipe = pipeline("object-detection", model="facebook/detr-resnet-50", device=0)

    @modal.method()
    def predict(self, image_bytes: bytes) -> list[dict]:
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.pipe(image)

class ModalBackend:
    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        # Programmatic client authentication logic is handled inside the CLI router
        import json
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with app.run():
            server = ModalModelServer()
            for img_file in input_path.glob("*.[jJ][pP][gG]"):
                img_bytes = img_file.read_bytes()
                predictions = server.predict.remote(img_bytes)
                
                out_file = output_path / f"{img_file.stem}_preds.json"
                with open(out_file, "w") as f:
                    json.dump(predictions, f, indent=4)

```

### Phase 5: Lightning Studio Jobs Architecture (`src/backends/lightning.py`)

Lightning Studio jobs provision cloud instances asynchronously. The CLI client will execute the SDK primitives to provision runtime resources, inject files across the secure storage layers, trigger executions, and subsequently synchronize outputs back down onto local systems.

```python
import os
from pathlib import Path
from src.backends.base import BaseInferenceBackend

class LightningJobsBackend(BaseInferenceBackend):
    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        from lightning_sdk import Studio, Machine, Job
        
        # Verify specific credentials are bound explicitly to avoid authentication bubbles
        user_id = os.getenv("LIGHTNING_USER_ID")
        api_key = os.getenv("LIGHTNING_API_KEY")
        
        # 1. Instantiate or reference a working Cloud Studio
        studio = Studio(name="inference-cluster-studio")
        studio.start()
        
        # 2. Synchronize local workspace artifacts to the cluster environment 
        # Programmatic copy maps directly to standard lit:// user-space allocations
        studio.upload(input_dir, remote_path="data/input_photos/")
        
        # 3. Launch an asynchronous containerized Batch Job on selected compute topologies
        job = Job.run(
            command=f"python -m src.backends.local --input data/input_photos/ --output data/output/ --model {model_id}",
            name="batch-detection-job",
            machine=Machine.T4,
            studio=studio
        )
        
        # Block client sequence execution thread waiting for job resolution status
        job.wait_for_status("SUCCEEDED")
        
        # 4. Pull down the final prediction JSON artifacts back to host machine
        studio.download("data/output/", local_path=output_dir)
        studio.stop()

```

### Phase 6: Core CLI Engine & User Orchestration Interactivity (`src/main.py`)

Construct the main CLI structure via `click`. This module enforces onboarding warnings and handles missing API tokens through interactive secure shell inputs, dynamically appending keys to an untracked local environment configuration file.

```python
import os
import click
from pathlib import Path
from dotenv import load_dotenv, set_key

# Target configuration storage isolated from development source code trees
CONFIG_DIR = Path.home() / ".config" / "object_detector"
ENV_FILE = CONFIG_DIR / ".env"

def init_environment():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch(mode=0o600)
    load_dotenv(ENV_FILE)

def interactive_credential_check(backend: str):
    init_environment()
    
    if backend == "lightning":
        if not os.getenv("LIGHTNING_API_KEY") or not os.getenv("LIGHTNING_USER_ID"):
            click.secho("\n=== LIGHTNING STUDIO AUTHENTICATION REQUIRED ===", fg="cyan", bold=True)
            click.echo("1. Log in via your web console interface at: https://lightning.ai")
            click.echo("2. Direct navigate to Settings -> API Keys to generate active credentials.")
            click.echo("3. Alternatively, execute 'lightning login' from your terminal wrapper.\n")
            
            uid = click.prompt("Enter your cloud LIGHTNING_USER_ID", type=str)
            key = click.prompt("Enter your cloud LIGHTNING_API_KEY", type=str, hide_input=True)
            
            set_key(str(ENV_FILE), "LIGHTNING_USER_ID", uid)
            set_key(str(ENV_FILE), "LIGHTNING_API_KEY", key)
            load_dotenv(ENV_FILE)

    elif backend == "modal":
        if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
            click.secho("\n=== MODAL LABS AUTHENTICATION REQUIRED ===", fg="cyan", bold=True)
            click.echo("1. Create an active development workspace profile via https://modal.com")
            click.echo("2. Initialize authentication configurations locally by executing: 'uv run modal setup'\n")
            
            token_id = click.prompt("Enter your MODAL_TOKEN_ID", type=str)
            token_secret = click.prompt("Enter your MODAL_TOKEN_SECRET", type=str, hide_input=True)
            
            set_key(str(ENV_FILE), "MODAL_TOKEN_ID", token_id)
            set_key(str(ENV_FILE), "MODAL_TOKEN_SECRET", token_secret)
            load_dotenv(ENV_FILE)

@click.command()
@click.option("--backend", type=click.Choice(["local", "lightning", "modal"]), required=True, help="Compute infrastructure target provider backend.")
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Path pointing to image folder directory.")
@click.option("--output-dir", type=click.Path(file_okay=False), default="./predictions", help="Local directory to write output JSON reports.")
@click.option("--model-id", type=str, default="facebook/detr-resnet-50", help="Hugging Face Model Hub Identifier string.")
def run_pipeline(backend, input_dir, output_dir, model_id):
    """Zero-Interaction Orchestration Execution Handler Client Routing Framework."""
    interactive_credential_check(backend)
    
    if backend == "local":
        from src.backends.local import LocalBackend
        engine = LocalBackend()
    elif backend == "lightning":
        from src.backends.lightning import LightningJobsBackend
        engine = LightningJobsBackend()
    elif backend == "modal":
        from src.backends.modal_backend import ModalBackend
        engine = ModalBackend()

    click.echo(f"Initializing computational workflow pipeline targeting runtime backend: {backend}")
    engine.execute(input_dir=input_dir, output_dir=output_dir, model_id=model_id)
    click.secho("Workflow successfully processed and concluded.", fg="green", bold=True)

if __name__ == "__main__":
    run_pipeline()

```

---

## 5. Security & Isolation Verification (`.gitignore`)

To ensure no authentication keys leak to any public GitHub repositories during development sessions in Cursor, verify that your `.gitignore` contains the following lines:

```text
# Local Virtual Environments
.venv/
.uv/

# Local configuration dumps containing secrets
.env
.env.*
~/.config/object_detector/

# Cache directories
__pycache__/
*.pyc
.chunks/

```

---

## 6. Execution Instructions for Cursor

1. Copy the tracking definition entries blocks into a fresh local project directory repository.
2. Initialize and lock environment tracking specs using terminal processing frames:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

```


3. Execute local operational checks to ensure logic alignment patterns work natively:
```bash
uv run python src/main.py --backend local --input-dir ./test_images/ --output-dir ./test_out/

```


4. Scale out to remote cloud backends seamlessly by altering your CLI execution target arguments; the script handles missing infrastructure bindings automatically via real-time console telemetry inputs.
