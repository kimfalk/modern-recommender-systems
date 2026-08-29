"""Feature engineering for the ordering stage (chapter 7).

Listing 7.1 shows the three family builders; this module is the full
runnable version, including the feature-column constants, the upstream
cross-encoder-score merge, and the frame assembly used by the notebook.

Design decisions mirrored from the chapter:
- The cross-encoder score from chapter 5 IS a feature (SCORE_COLS). The
  null hypothesis is the scored order, so the honest question is "how
  much on top", not pass/fail.
- FEATURE_COLS_NO_CROSS supports the cross-feature ablation in 7.6.1.
- DCN_DENSE_COLS excludes the hand-engineered crosses: DCN-v2 is
  supposed to learn them, so feeding it x_* would muddy the comparison.
- All builders must be called on the training window only (temporal
  cutoff discipline from chapter 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

ITEM_GENRE_COLS = [f"genre_{g}" for g in GENRES]
USER_AFF_COLS = [f"aff_{g}" for g in GENRES]
USER_STAT_COLS = [
    "user_n_ratings", "user_mean_rating", "user_rating_std",
    "user_days_since_active",
]
ITEM_STAT_COLS = ["item_n_ratings", "item_mean_rating", "item_log_pop", "item_age"]
CROSS_COLS = ["x_genre_affinity", "x_rating_gap", "x_pop_for_light_user"]
SCORE_COLS = ["cross_encoder_score"]

FEATURE_COLS = (
    USER_STAT_COLS + USER_AFF_COLS + ITEM_STAT_COLS + ITEM_GENRE_COLS
    + CROSS_COLS + SCORE_COLS
)
FEATURE_COLS_NO_CROSS = [c for c in FEATURE_COLS if c not in CROSS_COLS]
DCN_DENSE_COLS = FEATURE_COLS_NO_CROSS


def build_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Turn MovieLens movies.csv into movieId + year + 19 genre indicators."""
    out = movies[["movieId"]].copy()
    out["year"] = (
        movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    )
    split = movies["genres"].astype(str).str.split("|")
    for genre, col in zip(GENRES, ITEM_GENRE_COLS):
        out[col] = split.apply(lambda gs: float(genre in gs))
    return out


def build_user_features(
    train: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    like_threshold: float = 4.0,
) -> pd.DataFrame:
    """Listing 7.1: user statistics plus the genre-affinity taste vector."""
    stats = train.groupby("userId")["rating"].agg(
        user_n_ratings="count",
        user_mean_rating="mean",
        user_rating_std="std",
    )
    last_ts = train.groupby("userId")["timestamp"].max()
    stats["user_days_since_active"] = (train["timestamp"].max() - last_ts) / 86_400

    liked = train[train["rating"] >= like_threshold]
    affinity = (
        liked.merge(genre_matrix[["movieId"] + ITEM_GENRE_COLS], on="movieId")
        .groupby("userId")[ITEM_GENRE_COLS]
        .mean()
    )
    affinity.columns = USER_AFF_COLS
    return stats.join(affinity).fillna(0.0)


def build_item_features(
    train: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    current_year: int | None = None,
) -> pd.DataFrame:
    """Listing 7.1: item statistics plus genre indicators, indexed by movieId."""
    if current_year is None:
        year_max = genre_matrix["year"].max()
        current_year = int(year_max) if np.isfinite(year_max) else 2020

    stats = train.groupby("movieId")["rating"].agg(
        item_n_ratings="count",
        item_mean_rating="mean",
    )
    item = genre_matrix.set_index("movieId").join(stats)
    item["item_n_ratings"] = item["item_n_ratings"].fillna(0.0)
    item["item_mean_rating"] = item["item_mean_rating"].fillna(train["rating"].mean())
    item["item_log_pop"] = np.log1p(item["item_n_ratings"])
    age = (current_year - item["year"]).clip(lower=0)
    item["item_age"] = age.fillna(age.median() if np.isfinite(age.median()) else 0.0)
    return item[ITEM_STAT_COLS + ITEM_GENRE_COLS]


def build_cross_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Listing 7.1: hand-engineered feature products."""
    rows = rows.copy()
    aff = rows[USER_AFF_COLS].to_numpy(dtype=np.float32)
    genres = rows[ITEM_GENRE_COLS].to_numpy(dtype=np.float32)
    n_genres = genres.sum(axis=1)
    rows["x_genre_affinity"] = (aff * genres).sum(axis=1) / np.maximum(n_genres, 1.0)
    rows["x_rating_gap"] = rows["item_mean_rating"] - rows["user_mean_rating"]
    rows["x_pop_for_light_user"] = rows["item_log_pop"] / np.log1p(
        rows["user_n_ratings"] + 1.0
    )
    return rows


def attach_labels(candidates: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """Mark a candidate relevant=1 if the user interacted with it in the
    held-out window -- the same ground truth chapter 5 evaluated against."""
    pos = heldout[["userId", "movieId"]].drop_duplicates().assign(relevant=1)
    out = candidates.merge(pos, on=["userId", "movieId"], how="left")
    out["relevant"] = out["relevant"].fillna(0).astype(int)
    return out


def attach_upstream_scores(
    candidates: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    """Merge the chapter-5 cross-encoder scores onto the candidate frame.

    `scores` needs columns userId, movieId, cross_encoder_score. Every
    candidate must have a score: the scored order is the null hypothesis,
    so a missing score is a wiring bug, not something to impute away.
    """
    out = candidates.merge(
        scores[["userId", "movieId", "cross_encoder_score"]],
        on=["userId", "movieId"],
        how="left",
    )
    if out["cross_encoder_score"].isna().any():
        n = int(out["cross_encoder_score"].isna().sum())
        raise ValueError(
            f"{n} candidates have no cross_encoder_score; "
            "wire in the chapter-5 scorer output for the full candidate set."
        )
    return out


def build_feature_frame(
    candidates: pd.DataFrame,
    train: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    current_year: int | None = None,
) -> pd.DataFrame:
    """Assemble the full per-candidate feature frame.

    `candidates` must carry userId, movieId, cross_encoder_score and (for
    training/evaluation) relevant. Everything is computed from `train`
    only -- pass the training window, never the full ratings table.
    """
    users = build_user_features(train, genre_matrix)
    items = build_item_features(train, genre_matrix, current_year)
    rows = candidates.merge(users, left_on="userId", right_index=True, how="left")
    rows = rows.merge(items, left_on="movieId", right_index=True, how="left")
    fill_cols = USER_STAT_COLS + USER_AFF_COLS + ITEM_STAT_COLS + ITEM_GENRE_COLS
    rows[fill_cols] = rows[fill_cols].fillna(0.0)
    return build_cross_features(rows)
