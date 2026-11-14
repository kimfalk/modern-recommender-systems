"""One MLflow store for every chapter.

MLflow's default store is a ``mlflow.db`` in whatever directory the process
starts in, so each notebook folder used to grow its own database and the UI
only ever showed one chapter. ``setup_mlflow`` pins every notebook to the
same store at the repository root:

    tracking:   <repo>/mlflow.db           (runs, params, metrics)
    artifacts:  <repo>/mlartifacts/<name>  (logged files and models)

View everything with, from the repository root:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

Set ``MLFLOW_TRACKING_URI`` to send runs elsewhere instead (a tracking
server, or a Drive folder on Colab); then MLflow's own defaults apply.

Usage
-----
from recsys.utils.tracking import setup_mlflow
mlflow = setup_mlflow("chapter-05-two-tower")
"""

from __future__ import annotations

import os
from pathlib import Path

def _repo_root() -> Path:
    """The checkout this package lives in; the working directory when the
    package was pip-installed (e.g. on Colab) and there is no checkout."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "notebooks").is_dir() else Path.cwd()


REPO_ROOT = _repo_root()
TRACKING_DB = REPO_ROOT / "mlflow.db"
ARTIFACT_ROOT = REPO_ROOT / "mlartifacts"


def tracking_uri() -> str:
    """The shared store's URI (or MLFLOW_TRACKING_URI, if set)."""
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{TRACKING_DB}"


def setup_mlflow(experiment: str, verbose: bool = True):
    """Point MLflow at the shared store, select `experiment`, return `mlflow`.

    New experiments get their artifacts under <repo>/mlartifacts/<experiment>,
    so logged files land in one place no matter where the notebook runs.
    """
    import mlflow

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(ARTIFACT_ROOT / experiment).as_uri())
    mlflow.set_experiment(experiment)
    if verbose:
        print(f"MLflow: experiment '{experiment}' -> {uri}")
    return mlflow
