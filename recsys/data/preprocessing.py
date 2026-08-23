"""
recsys/data/preprocessing.py
============================
Shared preprocessing for the MovieLens interaction data: positive-interaction
filtering, user sampling, per-user temporal splits, and the item index
mappings that translate between the application's string item IDs and the
dense integer indices the models train on.

The ID contract used throughout the package:

  - Notebooks and pipeline stages speak **string item IDs** (as returned by
    ``recsys.data.loaders.load_movielens``).
  - Models and embedding tables speak **dense integer indices**.
  - The mappings built by :func:`build_item_index` are the only bridge, and
    they are built exactly once per experiment. Mixing the two spaces is one
    of the easiest data bugs to introduce and one of the hardest to spot,
    because everything still runs.

Usage
-----
from recsys.data.preprocessing import (
    filter_positive, sample_active_users, filter_min_item_ratings,
    temporal_split_per_user, build_item_index, user_item_lists, add_item_idx,
)
"""

from __future__ import annotations

import pandas as pd


def filter_positive(ratings: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """Keep only ratings at or above ``threshold`` -- the positive interactions."""
    return ratings[ratings["rating"] >= threshold].copy()


def sample_active_users(ratings: pd.DataFrame, n_users: int = 5_000,
                        min_ratings: int = 20, seed: int = 42) -> pd.DataFrame:
    """Sample up to ``n_users`` users that have at least ``min_ratings`` ratings.

    If fewer eligible users exist than requested (e.g. on a small dataset),
    all eligible users are kept.
    """
    counts = ratings.groupby("userId").size()
    eligible = counts[counts >= min_ratings].index
    if len(eligible) > n_users:
        sampled = pd.Series(eligible).sample(n_users, random_state=seed)
        return ratings[ratings["userId"].isin(sampled)].copy()
    return ratings[ratings["userId"].isin(eligible)].copy()


def filter_min_item_ratings(ratings: pd.DataFrame,
                            min_ratings: int = 30) -> pd.DataFrame:
    """Drop items with fewer than ``min_ratings`` ratings inside the sample."""
    counts = ratings.groupby("movieId").size()
    keep = counts[counts >= min_ratings].index
    return ratings[ratings["movieId"].isin(keep)].copy()


def temporal_split_per_user(interactions: pd.DataFrame, test_frac: float = 0.2,
                            min_interactions: int = 5):
    """Per-user temporal split: the most recent ``test_frac`` of each user's
    interactions form the test set (the Chapter 4 protocol).

    Users with fewer than ``min_interactions`` interactions stay entirely in
    the training set -- a one-item history split against a one-item test set
    measures noise, not quality.

    Returns
    -------
    (train_df, test_df) with the input columns preserved.
    """
    interactions = interactions.sort_values(["userId", "timestamp"])
    train_parts, test_parts = [], []
    for _, group in interactions.groupby("userId"):
        if len(group) < min_interactions:
            train_parts.append(group)
            continue
        cut = max(1, int(len(group) * (1 - test_frac)))     #A
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts) if test_parts else interactions.iloc[0:0]
    return train_df, test_df

#A At least one interaction always remains in training


def build_item_index(interactions: pd.DataFrame, movies: pd.DataFrame):
    """Build the bidirectional mapping between string item IDs and dense indices.

    Returns
    -------
    item_ids : list[str]
        Item IDs in index order -- position ``i`` is the ID of index ``i``.
    item_to_idx : dict[str, int]
    idx_to_title : dict[int, str]
    idx_to_genres : dict[int, list[str]]
        Genres split on the MovieLens ``|`` separator.
    """
    item_ids = sorted(interactions["movieId"].unique())
    item_to_idx = {mid: i for i, mid in enumerate(item_ids)}
    title_by_id = dict(zip(movies["movieId"], movies["title"]))
    genres_by_id = dict(zip(movies["movieId"], movies["genres"]))
    idx_to_title = {i: title_by_id.get(mid, f"movie {mid}")
                    for mid, i in item_to_idx.items()}
    idx_to_genres = {i: str(genres_by_id.get(mid, "")).split("|")
                     for mid, i in item_to_idx.items()}
    return item_ids, item_to_idx, idx_to_title, idx_to_genres


def user_item_lists(interactions: pd.DataFrame,
                    item_col: str = "movieId") -> dict:
    """Per-user chronological item lists, keyed by userId.

    ``item_col`` selects the ID space: ``"movieId"`` for string IDs (pipeline
    stages), or an index column such as ``"item_idx"`` for model training.
    """
    ordered = interactions.sort_values(["userId", "timestamp"])
    return ordered.groupby("userId")[item_col].apply(list).to_dict()


def add_item_idx(interactions: pd.DataFrame,
                 item_to_idx: dict) -> pd.DataFrame:
    """Attach the dense ``item_idx`` column used by the models."""
    out = interactions.copy()
    out["item_idx"] = out["movieId"].map(item_to_idx)
    return out
