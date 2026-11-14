"""Chapter 7: export the model-selection rows (the split shifted one slice back).

Fits the chapter-5 upstream models (two-tower retrieval + cross-encoder) on
each user's first 60% of positives and scores candidates for the users with
labels in the 60-70% slice. Writes, next to the existing exports:

  candidates_tune.parquet   candidates + cross-encoder scores for the tuning rows
  tuning_labels.parquet     the 60-70% positives (the tuning labels)

It starts from ch07_00_export_upstream.ipynb's outputs (positives.parquet,
movies.parquet), so the raw MovieLens files are not needed, and checks that
the slices and the item index reproduce the stored ones before fitting.
The notebook runs the same steps when it is rerun from the raw data.

    python scripts/ch07_export_tuning_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.data.preprocessing import build_item_index
from recsys.fourstage_recsys.ordering.splits import (
  per_user_temporal_slices,
  tuning_slices,
)
from recsys.fourstage_recsys.ordering.upstream import fit_upstream, score_candidates

ART = ROOT / "data" / "processed" / "chapter07"
SEED = 42  # the export notebook's seed
K_RETRIEVE = 100


def main() -> None:
  positives = pd.read_parquet(ART / "positives.parquet")
  movies = pd.read_parquet(ART / "movies.parquet")
  interactions = positives.drop(columns="slice")

  def keys(frame):
    return set(zip(frame.userId, frame.movieId, frame.timestamp))

  early, mid, late = per_user_temporal_slices(interactions, cuts=(0.7, 0.8))
  for name, part in (("early", early), ("mid", mid), ("late", late)):
    if keys(part) != keys(positives[positives["slice"] == name]):
      raise RuntimeError(f"recomputed {name} slice differs from positives.parquet")
  tune_early, tune_labels = tuning_slices(interactions)
  if keys(pd.concat([tune_early, tune_labels])) != keys(early):
    raise RuntimeError("tune_early + tune_labels is not the early slice")

  item_ids, item_to_idx, _, _ = build_item_index(interactions, movies)
  if not (interactions["movieId"].map(item_to_idx) == interactions["item_idx"]).all():
    raise RuntimeError("item index differs from the exported item_idx")
  print(
    f"tune_early {len(tune_early):,} | tune_labels {len(tune_labels):,} "
    f"({tune_labels.userId.nunique():,} users)",
    flush=True,
  )

  upstream = fit_upstream(
    tune_early, item_ids, item_to_idx, device="cpu", seed=SEED, verbose=True
  )
  candidates = score_candidates(
    upstream, sorted(tune_labels.userId.unique()), k_retrieve=K_RETRIEVE, device="cpu"
  )
  print(f"{len(candidates):,} candidate rows for {candidates.userId.nunique():,} users")
  candidates.to_parquet(ART / "candidates_tune.parquet", index=False)
  tune_labels.to_parquet(ART / "tuning_labels.parquet", index=False)
  print("wrote candidates_tune.parquet and tuning_labels.parquet")


if __name__ == "__main__":
  main()
