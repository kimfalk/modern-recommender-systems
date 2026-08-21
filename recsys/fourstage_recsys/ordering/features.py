"""Feature engineering for the ranking stage (chapter 7).

Builds the three feature families used by both rankers:

* user features   -- rating statistics, activity recency, genre affinities
* item features   -- popularity, average rating, age, genre indicators
* cross features  -- user-x-item interactions (genre affinity dot product,
                     rating gap)

Everything is computed from the *training window only*.  The public entry
points take the training interactions explicitly so that temporal leakage
is a type error rather than a silent bug: if you only have the train
DataFrame in scope, you cannot accidentally use test-period statistics.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# The 19 MovieLens genre labels ("(no genres listed)" is dropped).
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

USER_STAT_COLS = [
    "user_n_ratings",
    "user_mean_rating",
    "user_rating_std",
    "user_days_since_active",
]
USER_AFF_COLS = [f"user_aff_{g}" for g in GENRES]
ITEM_STAT_COLS = [
    "item_n_ratings",
    "item_mean_rating",
    "item_log_pop",
    "item_age",
]
ITEM_GENRE_COLS = [f"item_is_{g}" for g in GENRES]
CROSS_COLS = ["x_genre_affinity", "x_rating_gap", "x_pop_for_light_user"]

#: The columns both rankers train on, in a stable order.
FEATURE_COLS = (
    USER_STAT_COLS + USER_AFF_COLS + ITEM_STAT_COLS + ITEM_GENRE_COLS + CROSS_COLS
)


def build_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Return one row per movie with ``item_is_<genre>`` indicator columns
    and a ``year`` column parsed from the title.

    Parameters
    ----------
    movies:
        The MovieLens ``movies.csv`` frame (movieId, title, genres).
    """
    out = pd.DataFrame({"movieId": movies["movieId"].to_numpy()})
    genre_sets = movies["genres"].fillna("").str.split("|")
    for g in GENRES:
        out[f"item_is_{g}"] = genre_sets.apply(lambda s, g=g: int(g in s)).astype(
            np.int8
        )
    out["year"] = (
        movies["title"].astype(str).str.extract(_YEAR_RE, expand=False).astype(float)
    )
    return out


def build_user_features(
    train: pd.DataFrame, genre_matrix: pd.DataFrame, like_threshold: float = 4.0
) -> pd.DataFrame:
    """User-side features from the training window (listing 7.1).

    Returns a frame indexed by userId with rating statistics, recency, and
    a 19-dimensional genre-affinity vector: the mean genre indicators of
    the movies this user rated >= ``like_threshold``.
    """
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
    train: pd.DataFrame, genre_matrix: pd.DataFrame, current_year: int | None = None
) -> pd.DataFrame:
    """Item-side features from the training window (listing 7.1).

    Returns a frame indexed by movieId.  Items present in the catalog but
    unrated in the training window get zero counts rather than dropping
    out -- the ranker must be able to score cold candidates.
    """
    if current_year is None:
        current_year = int(
            pd.to_datetime(train["timestamp"].max(), unit="s").year
        )

    stats = train.groupby("movieId")["rating"].agg(
        item_n_ratings="count",
        item_mean_rating="mean",
    )
    item = genre_matrix.set_index("movieId").join(stats)
    item["item_n_ratings"] = item["item_n_ratings"].fillna(0.0)
    global_mean = train["rating"].mean()
    item["item_mean_rating"] = item["item_mean_rating"].fillna(global_mean)
    item["item_log_pop"] = np.log1p(item["item_n_ratings"])
    age = (current_year - item["year"]).clip(lower=0)
    item["item_age"] = age.fillna(age.median())  # ~1% of titles lack a year
    return item[ITEM_STAT_COLS + ITEM_GENRE_COLS]


def build_cross_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Pair features describing the user-item combination (listing 7.1).

    Expects ``rows`` to already carry the user and item feature columns.
    """
    aff = rows[USER_AFF_COLS].to_numpy(dtype=np.float32)
    genres = rows[ITEM_GENRE_COLS].to_numpy(dtype=np.float32)
    n_genres = genres.sum(axis=1)
    rows["x_genre_affinity"] = (aff * genres).sum(axis=1) / np.maximum(n_genres, 1.0)
    rows["x_rating_gap"] = rows["item_mean_rating"] - rows["user_mean_rating"]
    # Popularity matters more when we know little about the user.
    rows["x_pop_for_light_user"] = rows["item_log_pop"] / np.log1p(
        rows["user_n_ratings"] + 1.0
    )
    return rows


def assemble_examples(
    pairs: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join a (userId, movieId, ...) frame with all three feature families.

    Rows whose user or item never appeared in the training window are kept
    and filled with zeros -- again, the ranker has to survive cold inputs.
    """
    out = pairs.merge(user_features, left_on="userId", right_index=True, how="left")
    out = out.merge(item_features, left_on="movieId", right_index=True, how="left")
    out[USER_STAT_COLS + USER_AFF_COLS] = out[
        USER_STAT_COLS + USER_AFF_COLS
    ].fillna(0.0)
    out[ITEM_STAT_COLS + ITEM_GENRE_COLS] = out[
        ITEM_STAT_COLS + ITEM_GENRE_COLS
    ].fillna(0.0)
    return build_cross_features(out)
