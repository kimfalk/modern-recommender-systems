"""Temporal splits for the ordering stage (chapter 7).

The ordering model is a *stacked* model: it consumes the output of the
chapter-5 scorer as a feature. That creates two requirements the
single chapter-5 split cannot meet on its own:

1. The ranker needs training labels that are disjoint from its test
   labels, and both must come *after* the history its features see.
2. The upstream score on the ranker's training rows must come from a
   scorer that never saw those labels -- otherwise the ranker learns to
   trust an in-sample score and over-weights it at test time.

So each user's positive history is cut three ways, per user and in time
order, exactly as chapter 5 cuts it in two:

    [ early (0-70%) | mid (70-80%) | late (80-100%) ]

- Ranker TRAINING rows: upstream models fit on `early`, features from
  history before `mid`, labels = `mid`.
- Ranker TEST rows: upstream models fit on `early + mid` (= chapter 5's
  training set), features from history before `late`, labels = `late`
  (= chapter 5's test set).

The 80% cut reproduces `recsys.data.preprocessing.temporal_split_per_user`
row for row, so the test rows are graded on exactly chapter 5's held-out
set and the scored-order baseline is chapter 5's result.

Known limitation (also true of chapter 5): a per-user split is not a
global one. Item statistics computed from "history" include other users'
ratings that may be later in wall-clock time than this user's request.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_user_temporal_slices(
    interactions: pd.DataFrame,
    cuts: tuple[float, ...] = (0.7, 0.8),
    min_interactions: int = 5,
) -> list[pd.DataFrame]:
    """Cut each user's chronological history at the given fractions.

    Returns len(cuts) + 1 frames. Users with fewer than `min_interactions`
    interactions go entirely into the first slice, matching chapter 5.
    The last cut uses the same arithmetic as temporal_split_per_user, so
    slices[:-1] concatenated equals its train_df and slices[-1] its test_df.
    """
    df = interactions.sort_values(["userId", "timestamp"])     # same sort as ch5
    pos = df.groupby("userId").cumcount().to_numpy()
    n = df.groupby("userId")["userId"].transform("size").to_numpy()
    slice_id = np.zeros(len(df), dtype=int)
    eligible = n >= min_interactions
    for c in cuts:
        bound = np.maximum(1, np.floor(n * c).astype(int))     # == max(1, int(n*c))
        slice_id += (eligible & (pos >= bound)).astype(int)
    return [df[slice_id == s] for s in range(len(cuts) + 1)]


def feature_history(
    ratings: pd.DataFrame,
    future_positives: pd.DataFrame,
    like_threshold: float = 4.0,
) -> pd.DataFrame:
    """Everything a model may see when predicting `future_positives`.

    `ratings` holds all ratings (any value) for the sampled users/items.
    Removed: the future positives themselves, and every non-positive
    rating at or after the user's first future positive. Users with no
    future positives keep their full history.
    """
    keys = future_positives[["userId", "movieId"]].drop_duplicates().assign(_future=1)
    r = ratings.merge(keys, on=["userId", "movieId"], how="left")
    r = r[r["_future"].isna()].drop(columns="_future")
    boundary = r["userId"].map(request_times(future_positives))
    keep = boundary.isna() | (r["rating"] >= like_threshold) | (r["timestamp"] < boundary)
    return r[keep].reset_index(drop=True)


def request_times(future_positives: pd.DataFrame) -> pd.Series:
    """Per-user request time: the timestamp of the first held-out positive."""
    return future_positives.groupby("userId")["timestamp"].min()
