"""End-to-end smoke test for the chapter 7 ranking pipeline.

Generates synthetic data with the MovieLens schema (ratings.csv /
movies.csv) and genuine structure -- users have latent genre preferences
-- so both rankers should comfortably beat popularity if the pipeline is
wired correctly.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")

from recsys.fourstage_recsys.ranking import (
    FEATURE_COLS,
    GENRES,
    DCNv2Ranker,
    FeatureEmbedder,
    GroupedRankingDataset,
    MovieIndex,
    build_genre_matrix,
    build_item_features,
    build_user_features,
    compare_models,
    fit_dcn,
    make_ranking_dataset,
    popularity_scores,
    predict_scores,
    temporal_split,
    train_lambdamart,
)

rng = np.random.default_rng(7)

# --- synthetic MovieLens-schema data with learnable genre structure ------
N_USERS, N_MOVIES, N_RATINGS = 600, 800, 60_000

movie_genres = [
    "|".join(rng.choice(GENRES, size=rng.integers(1, 4), replace=False))
    for _ in range(N_MOVIES)
]
movies = pd.DataFrame(
    {
        "movieId": np.arange(1, N_MOVIES + 1),
        "title": [
            f"Movie {i} ({int(y)})"
            for i, y in enumerate(rng.integers(1960, 2020, N_MOVIES), start=1)
        ],
        "genres": movie_genres,
    }
)

# Latent taste: each user likes 2-3 genres; ratings reflect genre match
# plus an item quality term plus noise.
genre_ix = {g: i for i, g in enumerate(GENRES)}
user_pref = np.zeros((N_USERS + 1, len(GENRES)))
for u in range(1, N_USERS + 1):
    liked = rng.choice(len(GENRES), size=rng.integers(2, 4), replace=False)
    user_pref[u, liked] = 1.0
movie_vec = np.zeros((N_MOVIES + 1, len(GENRES)))
for m, gs in zip(movies["movieId"], movies["genres"]):
    for g in gs.split("|"):
        movie_vec[m, genre_ix[g]] = 1.0
quality = rng.normal(0, 0.5, N_MOVIES + 1)

pop_weights = np.exp(rng.normal(0, 1.2, N_MOVIES))  # long-tail popularity
pop_weights /= pop_weights.sum()

u_col = rng.integers(1, N_USERS + 1, N_RATINGS)
m_col = rng.choice(np.arange(1, N_MOVIES + 1), size=N_RATINGS, p=pop_weights)
match = (user_pref[u_col] * movie_vec[m_col]).sum(axis=1)
raw = 2.6 + 1.1 * match + quality[m_col] + rng.normal(0, 0.6, N_RATINGS)
ratings = pd.DataFrame(
    {
        "userId": u_col,
        "movieId": m_col,
        "rating": np.clip(np.round(raw * 2) / 2, 0.5, 5.0),
        "timestamp": rng.integers(1_000_000_000, 1_600_000_000, N_RATINGS),
    }
).drop_duplicates(["userId", "movieId"])
print(f"synthetic ratings: {len(ratings):,} rows")

# --- the pipeline, exactly as in the notebook ----------------------------
train, valid, test = temporal_split(ratings)
print(f"split: train {len(train):,} / valid {len(valid):,} / test {len(test):,}")

genre_matrix = build_genre_matrix(movies)
user_features = build_user_features(train, genre_matrix)
item_features = build_item_features(train, genre_matrix)
print(f"features: {len(FEATURE_COLS)} columns")
assert not user_features.isna().any().any()
assert not item_features.isna().any().any()

N_NEG = 50
frames = {}
data = {}
for name, window in [("train", train), ("valid", valid), ("test", test)]:
    frame, X, y, groups = make_ranking_dataset(
        window, train, user_features, item_features, n_negatives=N_NEG, seed=1
    )
    frames[name] = frame
    data[name] = (X, y, groups)
    assert groups.sum() == len(X)
    print(f"{name}: {len(X):,} rows, {len(groups):,} groups, "
          f"pos rate {(y > 0).mean():.3f}")

# --- LambdaMART -----------------------------------------------------------
t0 = time.time()
ranker = train_lambdamart(
    *data["train"], *data["valid"], num_boost_round=300, log_every=100
)
print(f"LambdaMART trained in {time.time() - t0:.1f}s, "
      f"best iter {ranker.best_iteration}")
lgb_scores = ranker.predict(data["test"][0])

# --- DCN-v2 ---------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
movie_index = MovieIndex(train["movieId"].unique())
embedder = FeatureEmbedder(
    {"movie": movie_index.cardinality}, n_dense=len(FEATURE_COLS), emb_dim=16
)
model = DCNv2Ranker(embedder, n_cross_layers=2, rank=32, deep_dims=(128, 64))
train_ds = GroupedRankingDataset(
    frames["train"], data["train"][2], movie_index, n_negatives=8
)
print(f"DCN training examples: {len(train_ds):,}, device: {device}")
t0 = time.time()
result = fit_dcn(
    model,
    train_ds,
    frames["valid"],
    data["valid"][1],
    data["valid"][2],
    movie_index,
    device,
    epochs=6,
    patience=2,
    batch_size=256,
)
print(f"DCN trained in {time.time() - t0:.1f}s, best epoch {result['best_epoch']}")
dcn_scores = predict_scores(model, frames["test"], movie_index, device)

# --- head-to-head ----------------------------------------------------------
report = compare_models(
    {
        "Popularity": popularity_scores(frames["test"]),
        "LambdaMART": lgb_scores,
        "DCN-v2": dcn_scores,
    },
    data["test"][1],
    data["test"][2],
)
print("\n", report.round(4))

pop = report.loc["Popularity", "ndcg@10"]
assert report.loc["LambdaMART", "ndcg@10"] > pop, "LambdaMART should beat popularity"
assert report.loc["DCN-v2", "ndcg@10"] > pop, "DCN-v2 should beat popularity"
print("\nSMOKE TEST PASSED")
