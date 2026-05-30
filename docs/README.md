# Remote job launch guides

Reference documentation for running GPU workloads from a local Python orchestrator.

| Guide | Service | Summary |
| --- | --- | --- |
| [launching-remote-jobs-with-modal.md](./launching-remote-jobs-with-modal.md) | [Modal Labs](https://modal.com) | Serverless `@app.cls` GPU workers, secrets, split local/remote modules |
| [launching-remote-jobs-with-lightning.md](./launching-remote-jobs-with-lightning.md) | [Lightning AI](https://lightning.ai) | GPU Studios, teamspace auth, upload/run/download lifecycle |

These guides mirror patterns implemented in `src/backends/modal_app.py`, `src/backends/modal_backend.py`, and `src/backends/lightning.py`.
