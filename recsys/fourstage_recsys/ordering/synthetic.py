"""Synthetic MovieLens-schema data for smoke tests and demo mode.

This stands in until the chapter-5 pipeline is wired in (open item:
real two-tower candidates + cross-encoder scores). Generated so the
scored order is a strong but beatable baseline: the synthetic
cross-encoder score sees the content affinity (with noise) but not the
behavioral/popularity signal, so a ranker that uses both has honest room
to add value on top -- mirroring the chapter's framing.

Everything is deterministic under `seed`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import GENRES


def make_synthetic_dataset(
    n_users: int = 60,
    n_items: int = 400,
    n_candidates: int = 40,
    seed: int = 7,
):
    """Returns (train, heldout, movies, candidates).

    - train / heldout: userId, movieId, rating, timestamp (temporal split)
    - movies: movieId, title, genres (MovieLens schema)
    - candidates: userId, movieId, relevant, cross_encoder_score
    """
    rng = np.random.default_rng(seed)
    n_genres = len(GENRES)

    n_clusters = 5
    cluster_of = [list(range(c, n_genres, n_clusters)) for c in range(n_clusters)]
    item_genres = np.zeros((n_items, n_genres))
    item_cluster = rng.integers(0, n_clusters, size=n_items)
    for i in range(n_items):
        pool = cluster_of[item_cluster[i]]
        k = int(rng.integers(1, 3))
        item_genres[i, rng.choice(pool, size=min(k, len(pool)), replace=False)] = 1.0
        if rng.random() < 0.1:
            item_genres[i, rng.integers(0, n_genres)] = 1.0
    item_norm = item_genres / np.maximum(
        np.linalg.norm(item_genres, axis=1, keepdims=True), 1e-9
    )
    years = rng.integers(1950, 2020, size=n_items)
    popularity = rng.zipf(1.6, size=n_items).astype(float)
    pop_z = (np.log1p(popularity) - np.log1p(popularity).mean())
    pop_z = pop_z / max(pop_z.std(), 1e-9)

    movies = pd.DataFrame({
        "movieId": np.arange(1, n_items + 1),
        "title": [f"Synthetic Movie {i} ({y})" for i, y in enumerate(years)],
        "genres": [
            "|".join(GENRES[j] for j in np.flatnonzero(item_genres[i]))
            for i in range(n_items)
        ],
    })

    user_cluster = rng.integers(0, n_clusters, size=n_users)
    user_pref = np.full((n_users, n_genres), 0.02)
    for u in range(n_users):
        user_pref[u, cluster_of[user_cluster[u]]] = 1.0
    user_pref *= rng.dirichlet(np.ones(n_genres) * 2.0, size=n_users) + 0.2
    user_pref /= user_pref.sum(axis=1, keepdims=True)
    utility = user_pref @ item_norm.T + 0.25 * pop_z[None, :]
    utility = utility + rng.normal(0, 0.15, size=utility.shape)

    rows = []
    ts0 = 1_500_000_000
    for u in range(n_users):
        n_rated = int(rng.integers(30, 70))
        rated = np.argsort(-utility[u])[: n_rated * 2]
        rated = rng.choice(rated, size=n_rated, replace=False)
        for j, item in enumerate(rated):
            base = 2.5 + 2.5 * (utility[u, item] - utility[u].min()) / (
                utility[u].max() - utility[u].min() + 1e-9
            )
            rating = float(np.clip(np.round(base * 2) / 2, 0.5, 5.0))
            rows.append((u + 1, int(item) + 1, rating,
                         ts0 + int(rng.integers(0, 80_000_000))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    cutoff = int(ratings["timestamp"].quantile(0.8))
    train = ratings[ratings["timestamp"] <= cutoff].reset_index(drop=True)
    heldout = ratings[ratings["timestamp"] > cutoff].reset_index(drop=True)

    cand_rows = []
    heldout_by_user = heldout.groupby("userId")["movieId"].apply(set).to_dict()
    train_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()
    for u in range(1, n_users + 1):
        positives = list(heldout_by_user.get(u, set()))[:6]
        if not positives:
            continue
        seen = train_by_user.get(u, set()) | set(positives)
        negatives = [m for m in range(1, n_items + 1) if m not in seen]
        neg = rng.choice(negatives, size=max(n_candidates - len(positives), 5),
                         replace=False)
        for m in positives:
            cand_rows.append((u, int(m), 1))
        for m in neg:
            cand_rows.append((u, int(m), 0))
    candidates = pd.DataFrame(cand_rows, columns=["userId", "movieId", "relevant"])

    aff = np.array([
        float(user_pref[u - 1] @ item_norm[m - 1])
        for u, m in zip(candidates["userId"], candidates["movieId"])
    ])
    aff_z = (aff - aff.mean()) / max(aff.std(), 1e-9)
    candidates["cross_encoder_score"] = (
        aff_z + rng.normal(0, 0.6, size=len(candidates))
    )
    return train, heldout, movies, candidates
