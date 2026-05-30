"""Abstract backend interface for inference execution."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseInferenceBackend(ABC):
    """Standardized execution schema for all compute targets."""

    @abstractmethod
    def execute(self, input_dir: str, output_dir: str, model_id: str) -> None:
        """Run object detection over images in ``input_dir`` and write JSON to ``output_dir``."""
