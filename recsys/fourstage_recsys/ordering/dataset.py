"""Tensor preparation for the neural ordering model (chapter 7, 7.5.2).

A tree eats a table; a network eats vectors. This module maps movieIds
to contiguous embedding indices, fits the dense-feature scaler on the
training window only (temporal-cutoff discipline: a scaler fit on all
the data leaks the future), and wraps candidate rows as a torch Dataset.
"""
from __future__ import annotations

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def make_movie_index(movie_ids) -> dict:
    """Map raw movieIds to contiguous indices for the embedding table."""
    return {m: i for i, m in enumerate(sorted(set(int(m) for m in movie_ids)))}


def add_movie_index(frame: pd.DataFrame, movie_index: dict) -> pd.DataFrame:
    out = frame.copy()
    out["movie_idx"] = out["movieId"].map(movie_index)
    if out["movie_idx"].isna().any():
        raise ValueError(
            "candidate frame contains movieIds missing from the movie index; "
            "build the index over the union of all candidate movieIds."
        )
    out["movie_idx"] = out["movie_idx"].astype(int)
    return out


def fit_scaler(train_frame: pd.DataFrame, dense_cols: list[str]) -> StandardScaler:
    """Fit the standardizer on the TRAINING window only."""
    return StandardScaler().fit(train_frame[dense_cols].to_numpy())


class CandidateDataset(Dataset):
    """Listing 7.5: scored candidates as tensors for the DCN-v2 ranker."""

    def __init__(self, frame: pd.DataFrame, dense_cols: list[str], scaler):
        self.movie = torch.as_tensor(frame["movie_idx"].to_numpy(copy=True))
        dense = scaler.transform(frame[dense_cols].to_numpy())
        self.dense = torch.as_tensor(dense, dtype=torch.float32)
        self.y = torch.as_tensor(
            frame["relevant"].to_numpy(copy=True), dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.movie[i], self.dense[i], self.y[i]


def make_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    dense_cols: list[str],
    scaler,
    batch_size: int = 1024,
):
    train_ds = CandidateDataset(train_frame, dense_cols, scaler)
    val_ds = CandidateDataset(val_frame, dense_cols, scaler)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
