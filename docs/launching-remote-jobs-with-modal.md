# Launching Remote GPU Jobs with Modal Labs

This guide explains how to run Python workloads on Modal serverless GPU infrastructure from a local orchestrator (CLI, script, or service). It is written for software engineers and AI coding agents integrating Modal into an existing codebase.

Modal is best understood as **ephemeral GPU containers** invoked from your laptop or CI. You define an `App`, package dependencies in an `Image`, expose methods on a `@app.cls` class, and call those methods with `.remote()` from a local client.

---

## 1. Mental model

| Concept | Role |
| --- | --- |
| **Local client** | Your machine: authenticates, reads inputs, calls Modal, writes outputs |
| **`modal.App`** | Named application grouping functions/classes |
| **`modal.Image`** | Container filesystem + Python dependencies built ahead of time |
| **`@app.cls`** | Stateful GPU worker (model loaded once per container) |
| **`@modal.enter()`** | Runs once when container starts (load weights) |
| **`@modal.method()`** | Callable entry point from local code via `.remote()` |
| **Secrets** | Inject env vars (e.g. `HF_TOKEN`) into remote containers |

Typical flow:

```
Local script
  → app.run()
  → ModalModelServer(model_id=...).predict.remote(payload)
  → GPU container executes predict()
  → result returned to local process
```

Cold start is usually under a few seconds after the first deploy; model loading dominates first inference.

---

## 2. Prerequisites

- Python $\geq$ 3.11
- Modal account: https://modal.com
- Package: `modal>=0.62.0`

Install and authenticate:

```bash
uv add modal
uv run modal setup   # writes credentials locally
```

Environment variables (alternative to `modal setup`):

| Variable | Purpose |
| --- | --- |
| `MODAL_TOKEN_ID` | API token id |
| `MODAL_TOKEN_SECRET` | API token secret |

Store secrets outside the repository (e.g. `~/.config/your_app/.env`, mode `0600`).

---

## 3. Project layout (recommended)

**Split remote app code from local orchestration.** Modal imports your module inside the container; heavy local-only imports (`python-dotenv`, `click`, etc.) will crash remote startup.

```
your_project/
├── src/
│   ├── remote/
│   │   └── modal_app.py      # ONLY what the container needs
│   └── client/
│       └── modal_runner.py     # local orchestration
```

### `modal_app.py` (runs in the cloud)

```python
"""Remote Modal app — keep imports minimal."""

import io

import modal

cuda_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "pillow>=10.3.0",
    "accelerate>=0.27.0",
    "timm>=0.9.0",  # required for some DETR checkpoints
)

app = modal.App("my-gpu-worker")


@app.cls(image=cuda_image, gpu="T4")
class GpuWorker:
    # Do NOT use `from __future__ import annotations` in this file.
    # Modal reads parameter types at runtime; postponed annotations break str types.

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
        return self.pipe(image)
```

### `modal_runner.py` (runs locally)

```python
"""Local client that invokes Modal."""

from pathlib import Path

import modal

from your_project.remote.modal_app import GpuWorker, app


def run_batch(image_paths: list[Path], model_id: str, hf_token: str | None) -> None:
    secrets = []
    if hf_token:
        secrets = [
            modal.Secret.from_dict(
                {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token}
            )
        ]

    with app.run():
        worker_cls = GpuWorker.with_options(secrets=secrets)
        worker = worker_cls(model_id=model_id)

        for path in image_paths:
            predictions = worker.predict.remote(path.read_bytes())
            # persist predictions locally...
```

---

## 4. Step-by-step implementation checklist

### Phase A — Define the remote app

1. Create `modal.Image` with **all** runtime dependencies (PyTorch, domain libraries, optional `timm`).
2. Define `modal.App("unique-app-name")`.
3. Implement `@app.cls(image=..., gpu="T4")` (or `A100`, etc.).
4. Load expensive state in `@modal.enter()`, not in `__init__`.
5. Expose work via `@modal.method()` with JSON-serializable I/O (bytes, dict, list).

### Phase B — Local orchestrator

1. Filter inputs locally (do not upload entire directories if they contain videos).
2. Open `with app.run():` before any `.remote()` calls.
3. Pass secrets for third-party APIs (Hugging Face, etc.).
4. Instantiate parameterized classes: `Worker(model_id="...")`.
5. Collect results and write artifacts locally.

### Phase C — Parallelism (recommended at scale)

Sequential `.remote()` in a loop works but underuses Modal. Prefer batching:

```python
image_bytes_list = [p.read_bytes() for p in image_paths]
for predictions in worker.predict.map(image_bytes_list):
    ...
```

Use `.map()` when images are independent and memory allows.

---

## 5. Authentication and secrets

| Approach | When to use |
| --- | --- |
| `modal setup` | Developer machines |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | CI, agents, headless scripts |
| `modal.Secret.from_dict({...})` | Hugging Face, API keys needed **inside** GPU container |

Secrets are attached per class invocation:

```python
worker_cls = GpuWorker.with_options(secrets=secrets)
worker = worker_cls(model_id=model_id)
```

---

## 6. Common pitfalls

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: dotenv` in remote logs | Local-only imports in `modal_app.py` | Split modules; remote file imports only cloud deps |
| `AttributeError: 'str' object has no attribute '__name__'` | `from __future__ import annotations` on `@app.cls` | Remove postponed annotations from modal app file |
| `InvalidError: modal.parameter() needs type-annotated` | Removed annotations entirely | Keep `model_id: str = modal.parameter(...)` but no `__future__` annotations |
| Job “failed” with exit tuple in error message | Misread return value | `.remote()` returns result; local `run_with_exit_code` is Lightning-specific |
| Uploading videos / junk files | `upload_folder` equivalent | Filter extensions locally; send bytes per file |
| Apple Silicon local testing | N/A on Modal | Modal GPU is always CUDA in container |

---

## 7. Minimal end-to-end script template

```python
#!/usr/bin/env python3
"""Run a batch job on Modal from the command line."""

import json
import os
from pathlib import Path

from myapp.remote.modal_app import GpuWorker, app

INPUT_DIR = Path("./inputs")
OUTPUT_DIR = Path("./outputs")
MODEL_ID = "facebook/detr-resnet-50"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = list_images(INPUT_DIR)
    if not images:
        raise SystemExit(f"No images found in {INPUT_DIR}")

    secrets = []
    if token := os.getenv("HF_TOKEN"):
        import modal
        secrets = [modal.Secret.from_dict({"HF_TOKEN": token})]

    with app.run():
        worker = GpuWorker.with_options(secrets=secrets)(model_id=MODEL_ID)
        for path in images:
            preds = worker.predict.remote(path.read_bytes())
            out = OUTPUT_DIR / f"{path.stem}_preds.json"
            out.write_text(json.dumps(preds, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

Deploy / run:

```bash
uv run python scripts/run_modal_batch.py
```

---

## 8. Verification checklist

- [ ] `modal setup` or token env vars present
- [ ] Remote module imports succeed inside container (no local CLI deps)
- [ ] First `.remote()` call completes; GPU visible in Modal dashboard
- [ ] Outputs appear locally with expected schema
- [ ] Non-image files in input directory are ignored
- [ ] Optional: `.map()` used for large batches

---

## 9. Guidance for AI coding agents

When implementing Modal support in a new repository:

1. **Create `remote/modal_app.py` first** with only cloud dependencies.
2. **Never import** credential helpers that pull `python-dotenv` into the Modal module.
3. **Use `@modal.enter()`** for model weights, not `__enter__` (deprecated).
4. **Parameterize** model id via `modal.parameter()`, not closure globals.
5. **Attach secrets** for Hugging Face when using gated or rate-limited models.
6. **Filter inputs** before `.remote()`; pass bytes or paths explicitly.
7. **Prefer `.map()`** over per-file sequential `.remote()` when batch size $> 1$.
8. **Test** with one small image before scaling to full datasets.

---

## 10. References

- Modal docs: https://modal.com/docs
- Lifecycle hooks (`@modal.enter`): https://modal.com/docs/guide/lifecycle-functions
- Parametrized classes: https://modal.com/docs/guide/parametrized-functions
- This repository: `src/backends/modal_app.py`, `src/backends/modal_backend.py`
