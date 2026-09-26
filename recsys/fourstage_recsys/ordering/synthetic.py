"""Synthetic MovieLens-schema data for smoke tests and demo mode.

Produces raw `ratings` and `movies` frames in the exact schema
recsys.data.loaders.load_movielens returns (string ids, unix timestamps),
so the synthetic run goes through the same code path as MovieLens 25M:
chapter-5 preprocessing -> three-way split -> upstream fit -> features.

Users have a genre taste, items have a genre mix, a release year and a
popularity; users pick items by taste + popularity + a preference for
items released around the time they are rating. Everything is
deterministic under `seed`. The numbers it produces mean nothing; it
exists so the notebooks and tests run end to end in minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import GENRES

T0 = 946_684_800          # 2000-01-01
YEAR = 365 * 86_400


def make_synthetic_movielens(n_users: int = 400, n_items: int = 300, seed: int = 7):
    """Returns (ratings, movies) in MovieLens schema."""
    rng = np.random.default_rng(seed)
    n_genres = len(GENRES)

    item_genres = np.zeros((n_items, n_genres))
    for i in range(n_items):
        item_genres[i, rng.choice(n_genres, size=int(rng.integers(1, 4)), replace=False)] = 1
    years = rng.integers(1990, 2020, size=n_items)
    pop = np.log1p(rng.zipf(1.8, size=n_items).astype(float))
    movies = pd.DataFrame({
        "movieId": [str(i + 1) for i in range(n_items)],
        "title": [f"Synthetic Movie {i + 1} ({y})" for i, y in enumerate(years)],
        "genres": ["|".join(GENRES[j] for j in np.flatnonzero(item_genres[i]))
                   for i in range(n_items)],
    })

    rows = []
    for u in range(n_users):
        taste = rng.dirichlet(np.full(n_genres, 0.3))
        start = T0 + rng.integers(0, 15 * YEAR)
        span = rng.integers(YEAR // 4, 4 * YEAR)
        n = int(rng.integers(25, 80))
        ts = np.sort(start + rng.integers(0, span, size=n))
        seen = set()
        for t in ts:
            year_now = 2000 + (t - T0) / YEAR
            fresh = -0.15 * np.abs(years - year_now)
            logits = 3 * (item_genres @ taste) + pop + fresh
            logits[list(seen)] = -np.inf
            p = np.exp(logits - logits.max()); p /= p.sum()
            i = int(rng.choice(n_items, p=p))
            seen.add(i)
            match = item_genres[i] @ taste / max(item_genres[i].sum(), 1)
            rating = np.clip(np.round((3.0 + 6 * match + rng.normal(0, 0.8)) * 2) / 2, 0.5, 5.0)
            rows.append((str(u + 1), str(i + 1), float(rating), int(t)))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return ratings, movies
