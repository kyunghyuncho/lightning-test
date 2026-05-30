# Object Detector CLI

A production-grade Python command-line interface for orchestrating object detection across three compute backends: **local machine** (CPU/MPS/CUDA), **Lightning Studio Jobs**, and **Modal Labs** serverless GPU inference.

## Architecture

The CLI routes inference requests through a unified backend interface. Each backend implements the same `execute(input_dir, output_dir, model_id)` contract while adapting to its target runtime semantics.

```
                                    +--------------> [ Local Compute (CPU/MPS) ]
                                    |
[ CLI Engine ] === (Backend Router) +--------------> [ Lightning Studio Job (SDK API) ]
                                    |
                                    +--------------> [ Modal Labs Serverless (gRPC) ]
```

| Feature / Backend | Local Machine | Lightning Studio Jobs | Modal Labs |
| --- | --- | --- | --- |
| Compute Semantics | Persistent local loops | Asynchronous batch job | Ephemeral serverless function |
| Hardware Target | macOS (`mps` / `cpu`) | Cloud GPU (e.g., `T4`, `A10G`) | On-demand GPU (e.g., `T4`, `A100`) |
| Data Transport | Direct disk I/O | Programmatic SDK sync | Ephemeral network payload |
| Cold Start Latency | $0$ seconds | $2$–$5$ minutes (provisioning) | $<2$ seconds |

## Requirements

- Python $\geq 3.11$
- [uv](https://github.com/astral-sh/uv) for environment management

## Installation

```bash
uv venv --python 3.11
source .venv/bin/activate
uv lock --exclude-newer "1 week"
uv sync --exclude-newer "1 week"
uv pip install -e .
```

## Credentials

Secrets are stored outside the repository at `~/.config/object_detector/.env` (mode `0600`).

| Backend | Environment Variables |
| --- | --- |
| Lightning AI | `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY`, `LIGHTNING_TEAMSPACE`, `LIGHTNING_USERNAME` (or `LIGHTNING_ORG`), `LIGHTNING_MACHINE` (default: `T4`) |
| Modal Labs | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| Hugging Face (optional) | `HF_TOKEN` |

The CLI prompts interactively for missing cloud credentials on first use. For Lightning, it also tries to auto-discover your teamspace from the API; if that fails, you'll be asked for your teamspace name and username. Inference runs directly on a GPU Studio (default `T4`), which should appear as the machine type in the Lightning console.

## Usage

Run interactively (prompts for backend, paths, and model):

```bash
uv run object-detector
```

Or pass flags directly for scripted runs:

```bash
uv run object-detector \
  --backend local \
  --input-dir ./test_images/ \
  --output-dir ./predictions/
```

Force the interactive wizard even when all flags are supplied:

```bash
uv run object-detector --interactive
```

After a run completes, the CLI prints a sample summary of detected objects per image (disable with `--no-show-samples`).

### Tab completion

Enable shell tab completion for flags (including folder paths for `--input-dir` and `--output-dir`):

```bash
# zsh (add to ~/.zshrc)
eval "$(_OBJECT_DETECTOR_COMPLETE=zsh_source object-detector)"

# bash (add to ~/.bashrc)
eval "$(_OBJECT_DETECTOR_COMPLETE=bash_source object-detector)"
```

Then restart your shell or `source` the rc file. Tab-complete after `--input-dir ` or `--output-dir ` to browse folders.

In interactive mode, folder path prompts support tab completion via `prompt_toolkit` (press Tab while typing a path).

### Local inference

```bash
uv run python src/main.py \
  --backend local \
  --input-dir ./test_images/ \
  --output-dir ./predictions/
```

Or via the installed entry point:

```bash
uv run object-detector \
  --backend local \
  --input-dir ./test_images/ \
  --output-dir ./predictions/
```

### Lightning Studio Jobs

```bash
uv run object-detector \
  --backend lightning \
  --input-dir ./test_images/ \
  --output-dir ./predictions/
```

### Modal Labs serverless

```bash
uv run object-detector \
  --backend modal \
  --input-dir ./test_images/ \
  --output-dir ./predictions/
```

## Project Layout

```
src/
├── main.py                # Click CLI entrypoint
├── core/
│   └── detector.py        # Hugging Face DETR/YOLOS pipeline
└── backends/
    ├── base.py            # Abstract backend interface
    ├── local.py           # Local inference
    ├── lightning.py       # Lightning Studio Jobs
    └── modal_backend.py   # Modal serverless GPU
tests/                     # Unit tests (mocked inference)
```

## Development

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run pytest tests/
```

## License

MIT
