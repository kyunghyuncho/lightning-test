# Launching Remote GPU Jobs with Lightning AI Studios

This guide explains how to run Python workloads on Lightning AI GPU **Studios** from a local orchestrator (CLI, script, or service). It is written for software engineers and AI coding agents integrating the `lightning-sdk` into an existing stack.

Lightning offers two related patterns:

1. **Studio (interactive GPU machine)** — persistent environment; upload files, run shell commands, download results. Best when your job is “run this Python script on a GPU VM.”
2. **Jobs plugin / `Job.run`** — asynchronous job from a Studio snapshot on a possibly different machine type. Best for long-running batch pipelines decoupled from the Studio UI.

This repository uses the **GPU Studio** pattern because it is easier to verify in the console and matches a synchronous CLI workflow.

---

## 1. Mental model

| Concept | Role |
| --- | --- |
| **Teamspace** | Project namespace (required for every API call) |
| **Owner** | User or organization that owns the teamspace |
| **Studio** | Cloud development runtime (files + terminal + GPU) |
| **`studio.start(Machine.T4)`** | Boots Studio on a GPU machine type |
| **`studio.upload_file` / `download_folder`** | Data plane between local disk and Studio |
| **`studio.run` / `run_with_exit_code`** | Execute shell commands on the Studio |
| **`studio.set_env`** | Push environment variables (dict, not key/value pairs) |

URL pattern:

```
https://lightning.ai/<owner>/<teamspace>/studios/<studio-name>/...
```

Typical flow:

```
Local script
  → Studio(name=..., teamspace=..., user=...).start(Machine.T4)
  → upload inputs + runner script
  → pip install dependencies on Studio
  → run_with_exit_code("python remote_script.py")
  → download_folder(outputs)
  → studio.stop()
```

Expect **minutes** for first Studio provisioning; subsequent starts are faster.

---

## 2. Prerequisites

- Python $\geq$ 3.11
- Lightning account: https://lightning.ai
- Package: `lightning-sdk>=0.3.0`

Install:

```bash
uv add lightning-sdk
```

### Required credentials

| Variable | Purpose |
| --- | --- |
| `LIGHTNING_USER_ID` | Programmatic user id (Settings → API Keys) |
| `LIGHTNING_API_KEY` | Programmatic API key |
| `LIGHTNING_TEAMSPACE` | Teamspace name from console URL |
| `LIGHTNING_USERNAME` | Owner username (if user-owned teamspace) |
| `LIGHTNING_ORG` | Organization name (if org-owned; use instead of username) |

Optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LIGHTNING_MACHINE` | `T4` | GPU type (`Machine.T4`, `T4_X_2`, `A100`, …) |
| `LIGHTNING_STUDIO_NAME` | `inference-cluster-studio` | Studio identifier |
| `HF_TOKEN` | — | Hugging Face Hub token on Studio |

Example `~/.config/your_app/.env`:

```bash
LIGHTNING_USER_ID=...
LIGHTNING_API_KEY=...
LIGHTNING_TEAMSPACE=Vision-model
LIGHTNING_USERNAME=Kc119
LIGHTNING_MACHINE=T4
HF_TOKEN=hf_...   # optional
```

---

## 3. Resolving teamspace and owner (critical)

`Studio(...)` **requires** a teamspace and owner. API keys alone are insufficient.

```python
from lightning_sdk import Studio

# Minimal explicit construction:
studio = Studio(
    name="my-inference-studio",
    teamspace="Vision-model",
    user="Kc119",          # or org="MyOrg" for org-owned teamspaces
    create_ok=True,
)
```

Auto-discovery via SDK (when credentials are valid):

```python
from lightning_sdk.utils.resolve import _get_authed_user, _get_teamspace_names_for_authed_user

username = _get_authed_user().name
teamspaces = _get_teamspace_names_for_authed_user()
```

If you see:

```
ValueError: Couldn't resolve teamspace from the provided name, org, or user
```

set `LIGHTNING_TEAMSPACE` and `LIGHTNING_USERNAME` (or `LIGHTNING_ORG`) explicitly.

---

## 4. Project layout (recommended)

```
your_project/
├── src/
│   ├── lightning_config.py   # teamspace + machine resolution
│   └── lightning_runner.py   # orchestration
└── remote/
    └── run_job.py            # script uploaded to Studio (optional inline generation)
```

---

## 5. Reference implementation

### 5.1 Build and start a GPU Studio

```python
import os
from lightning_sdk import Machine, Studio

def build_studio() -> Studio:
    return Studio(
        name=os.getenv("LIGHTNING_STUDIO_NAME", "inference-cluster-studio"),
        teamspace=os.environ["LIGHTNING_TEAMSPACE"],
        user=os.environ["LIGHTNING_USERNAME"],
        create_ok=True,
    )

def resolve_machine():
    name = os.getenv("LIGHTNING_MACHINE", "T4").upper()
    return getattr(Machine, name)

studio = build_studio()
machine = resolve_machine()
studio.start(machine)   # IMPORTANT: pass Machine.T4, not CPU default
assert "T4" in str(studio.machine)  # verify in console
```

**Do not** call `studio.start()` without `machine=` unless you intend to pay for a CPU Studio. The Lightning UI will show “4 x CPU” instead of “T4.”

### 5.2 Environment variables on the Studio

`set_env` expects a **dictionary**:

```python
# WRONG — raises ValueError
studio.set_env("HF_TOKEN", token)

# CORRECT
studio.set_env({"HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token})
```

### 5.3 Upload only intended files

**Do not** use `upload_folder` on a mixed directory (images + `.mp4` + logs). Upload per file:

```python
REMOTE_INPUT = "data/input_photos"

for image_path in image_files:
    studio.upload_file(
        str(image_path),
        remote_path=f"{REMOTE_INPUT}/{image_path.name}",
    )
```

Filter locally:

```python
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

def list_images(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
```

### 5.4 Install dependencies, then run

Order matters:

```python
studio.run("mkdir -p data/input_photos data/output")
# upload files...
studio.run(
    "pip install torch transformers pillow accelerate timm"
)
output, exit_code = studio.run_with_exit_code("python data/run_detection.py")
```

`run_with_exit_code` returns **`(stdout_and_stderr: str, exit_code: int)`**, not an int alone:

```python
if exit_code != 0:
    raise RuntimeError(f"Remote failed ({exit_code}):\n{output}")
```

A exit code of `0` with warnings in `output` is still success (e.g. Hugging Face unauthenticated download notice).

### 5.5 Remote runner script (on Studio)

```python
# data/run_detection.py (lives on Studio after upload)
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline

input_dir = Path("data/input_photos")
output_dir = Path("data/output")
output_dir.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this Studio machine.")

token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
pipe = pipeline("object-detection", model="facebook/detr-resnet-50", device=0, token=token)

for img in sorted(input_dir.iterdir()):
    if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    preds = pipe(Image.open(img).convert("RGB"))
    (output_dir / f"{img.stem}_preds.json").write_text(
        json.dumps(preds, indent=2), encoding="utf-8"
    )
```

### 5.6 Download results and stop

```python
studio.download_folder("data/output", target_path="./local_predictions")
studio.stop()
```

Always `stop()` in `finally` to avoid leaving a billable Studio running.

---

## 6. Alternative: Jobs plugin (async batch)

Use when you need a **separate** job machine or long-running async execution:

```python
from lightning_sdk import Machine

studio.install_plugin("jobs")
job = studio.installed_plugins["jobs"].run(
    command="python data/run_detection.py",
    name="batch-detection-job",
    machine=Machine.T4,
)
job.wait_for_status("SUCCEEDED")  # or job.wait() depending on SDK version
```

Notes:

- The **Studio** may remain on CPU while the **Job** uses T4 — check the **Jobs** tab, not only Studios.
- Dependencies must exist in the Studio snapshot or be installed in the job command.
- For simpler CLIs, prefer `studio.start(Machine.T4)` + `run_with_exit_code`.

---

## 7. Complete local orchestrator template

```python
#!/usr/bin/env python3
"""Run a GPU script on Lightning Studio."""

import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from lightning_sdk import Machine, Studio

load_dotenv(os.path.expanduser("~/.config/your_app/.env"))

INPUT_DIR = Path("./test_images")
OUTPUT_DIR = Path("./predictions")
REMOTE_SCRIPT = "data/run_job.py"
MODEL_ID = "facebook/detr-resnet-50"


def list_images(folder: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)


def main() -> None:
    images = list_images(INPUT_DIR)
    if not images:
        raise SystemExit("No images found")

    studio = Studio(
        name="inference-cluster-studio",
        teamspace=os.environ["LIGHTNING_TEAMSPACE"],
        user=os.environ["LIGHTNING_USERNAME"],
        create_ok=True,
    )
    machine = getattr(Machine, os.getenv("LIGHTNING_MACHINE", "T4"))

    studio.start(machine)
    try:
        if token := os.getenv("HF_TOKEN"):
            studio.set_env({"HF_TOKEN": token})

        studio.run("mkdir -p data/input_photos data/output")
        for img in images:
            studio.upload_file(str(img), remote_path=f"data/input_photos/{img.name}")

        script = textwrap.dedent(f"""
            # ... same as section 5.5, with model={MODEL_ID!r}
        """).strip()
        Path(".tmp_runner.py").write_text(script, encoding="utf-8")
        studio.upload_file(".tmp_runner.py", remote_path=REMOTE_SCRIPT)
        Path(".tmp_runner.py").unlink()

        studio.run("pip install torch transformers pillow accelerate timm")
        output, code = studio.run_with_exit_code(f"python {REMOTE_SCRIPT}")
        if code != 0:
            raise RuntimeError(output)
        studio.download_folder("data/output", target_path=str(OUTPUT_DIR))
    finally:
        studio.stop()


if __name__ == "__main__":
    main()
```

---

## 8. Common pitfalls

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Couldn't resolve teamspace` | Missing teamspace/owner | Set `LIGHTNING_TEAMSPACE` + `LIGHTNING_USERNAME` |
| Studio shows **4 x CPU** | `studio.start()` without GPU | `studio.start(Machine.T4)` |
| `dictionary update sequence element` on `set_env` | Passed two strings | `set_env({"KEY": "value"})` |
| False failure with long HF log + `0` | Tuple unpack bug | `output, exit_code = run_with_exit_code(...)` |
| Videos uploaded | `upload_folder` on mixed dir | Per-file `upload_file` after filtering |
| CUDA not available in script | Started CPU studio | Start GPU machine; check `nvidia-smi` via `studio.run` |
| Stale `.lightning_runner.py` in cwd | Writing temp script locally | Use `tempfile.NamedTemporaryFile` |

---

## 9. Verification checklist

- [ ] `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` set
- [ ] `LIGHTNING_TEAMSPACE` and owner resolve correctly
- [ ] Console shows Studio on **T4** (or chosen GPU)
- [ ] `studio.run("nvidia-smi")` succeeds
- [ ] Only image files uploaded
- [ ] `run_with_exit_code` returns exit code `0`
- [ ] `download_folder` populates local output directory
- [ ] `studio.stop()` runs in `finally`

---

## 10. Guidance for AI coding agents

When adding Lightning to a codebase:

1. **Never** construct `Studio(name=..., create_ok=True)` without `teamspace` and `user`/`org`.
2. **Always** call `studio.start(Machine.<GPU>)` for inference workloads.
3. **Use** `set_env(dict)` for secrets.
4. **Unpack** `(output, exit_code)` from `run_with_exit_code`.
5. **Upload** filtered files only; never blind `upload_folder` on user directories.
6. **Install** Python deps on the Studio **before** running the main script.
7. **Stop** the Studio in a `finally` block.
8. **Log** `studio.machine` after start so users can confirm GPU in the UI.
9. **Test** with one image before batching hundreds.

---

## 11. References

- Lightning SDK on PyPI: https://pypi.org/project/lightning-sdk/
- Lightning console: https://lightning.ai
- This repository: `src/backends/lightning.py`, `src/core/lightning_config.py`
