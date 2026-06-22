"""Optional experiment tracking via MLflow.

MLflow is an optional dependency. This wrapper is a no-op when MLflow is not
installed or tracking is disabled, so the pipeline never depends on it. When
available, every training run logs its params, metrics, and artifacts to a
local file store under ``mlruns/``.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

MLFLOW_AVAILABLE: bool = find_spec("mlflow") is not None


class ExperimentTracker:
    """Context manager that logs to MLflow if available, else does nothing."""

    def __init__(self, experiment_name: str, *, enabled: bool = True) -> None:
        self._active = enabled and MLFLOW_AVAILABLE
        self._experiment = experiment_name
        self._mlflow: Any = None
        if enabled and not MLFLOW_AVAILABLE:
            logger.info("MLflow not installed; experiment tracking disabled.")

    def __enter__(self) -> ExperimentTracker:
        if self._active:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_experiment(self._experiment)
            mlflow.start_run()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        if self._active:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._active:
            self._mlflow.log_metrics(metrics)

    def log_artifacts(self, directory: str | Path) -> None:
        if self._active:
            self._mlflow.log_artifacts(str(directory))
