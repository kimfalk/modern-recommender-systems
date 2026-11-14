"""One folder for every figure in the book.

Every notebook saves its book figures with ``save_figure``, so they all land
in ``<repo>/figures/chapter-NN/`` no matter where the notebook runs:

    figures/
      chapter-05/hard_negative_band.png
      chapter-05/hard_negative_band.svg
      chapter-14/ab_test_confidence_intervals.png
      ...

Each figure is written as a 300 dpi PNG and an SVG (vector, scales for
print), the two formats Manning asks for.

Usage
-----
from recsys.utils.figures import save_figure
save_figure(fig, "hard_negative_band", chapter=5)
"""

from __future__ import annotations

from pathlib import Path

from .paths import FIGURES_DIR


def chapter_dir(chapter: int | str) -> Path:
    """figures/chapter-NN for chapter 5 or "chapter-05"; created if missing."""
    name = f"chapter-{int(chapter):02d}" if str(chapter).isdigit() else str(chapter)
    path = FIGURES_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, name: str, chapter: int | str,
                formats: tuple = ("png", "svg"), dpi: int = 300,
                verbose: bool = True) -> list[Path]:
    """Save a matplotlib figure as figures/chapter-NN/<name>.<format>.

    Call it before plt.show() (or plt.close()): a figure that has been shown
    in a notebook is usually blank by the time it is saved.
    """
    out = chapter_dir(chapter)
    paths = [out / f"{name}.{ext}" for ext in formats]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if verbose:
        print(f"Saved figure: {out / name}.{{{','.join(formats)}}}")
    return paths
