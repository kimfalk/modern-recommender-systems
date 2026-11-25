"""Repository-level locations shared by every chapter."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """The checkout this package lives in; the working directory when the
    package was pip-installed (e.g. on Colab) and there is no checkout."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "notebooks").is_dir() else Path.cwd()


REPO_ROOT = _repo_root()
FIGURES_DIR = REPO_ROOT / "figures"


def display_path(path: Path | str) -> str:
    """`path` relative to the repository root, for printing.

    Notebook outputs are committed, so printed paths should read the same on
    every machine instead of exposing the author's home directory.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
