"""Where does each ranker put the first relevant item? (chapter 7).

NDCG@10 averages over users; the rank-bucket view shows where the lift
comes from -- more users with a hit in slot 1, or hits pulled up from
deep in the list into the top 10. first_relevant_rank gives the raw
per-user ranks; plot_rank_buckets draws them as shares per bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import ranks_within_user

DEFAULT_BUCKETS = ((1, 1), (2, 3), (4, 10), (11, 30), (31, None))


def first_relevant_rank(frame: pd.DataFrame, scores: np.ndarray) -> pd.Series:
  """1-based rank of each user's first relevant candidate; NaN if none.

  Ties are broken as in evaluate_ranker, so rank 1 here is the item the
  ranker's list shows first.
  """
  r = ranks_within_user(frame, scores)
  hits = r[r["relevant"] > 0].groupby("userId")["rank"].min() + 1
  users = pd.Index(np.unique(r["userId"].to_numpy()), name="userId")
  return hits.reindex(users).astype(float).rename("first_relevant_rank")


def bucket_shares(ranks: pd.Series, buckets=DEFAULT_BUCKETS) -> pd.Series:
  """Share of users whose first relevant item falls in each rank bucket."""
  shares = {}
  for lo, hi in buckets:
    label = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
    inside = ranks >= lo if hi is None else ranks.between(lo, hi)
    shares[label] = float(inside.mean())
  shares["none"] = float(ranks.isna().mean())
  return pd.Series(shares)


def plot_rank_buckets(ranks_by_model: dict, buckets=DEFAULT_BUCKETS, ax=None):
  """Grouped bars: share of users per first-relevant-rank bucket, per model."""
  import matplotlib.pyplot as plt

  table = pd.DataFrame(
    {name: bucket_shares(r, buckets) for name, r in ranks_by_model.items()}
  )
  if ax is None:
    _, ax = plt.subplots(figsize=(8, 4))
  table.plot.bar(ax=ax, rot=0, width=0.8)
  ax.set_xlabel("rank of first relevant item")
  ax.set_ylabel("share of users")
  ax.legend(fontsize=8)
  return ax, table
