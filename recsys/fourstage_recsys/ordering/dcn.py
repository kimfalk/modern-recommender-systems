"""DCN-v2 ordering model (chapter 7, section 7.5).

The cross network builds the multiplication into the architecture: each
layer computes x_{l+1} = x0 * (W_l x_l + b_l) + x_l, always multiplying
against the ORIGINAL input, so L layers give interactions up to degree
L+1. The deep MLP runs in parallel and catches the smooth structure the
crosses miss.

This is a network trained with a POINTWISE loss (BCE on the relevance
label) -- the mirror image of LambdaMART's tree-plus-listwise pairing.
The chapter shows no training-loop listing; train_dcn below is the
repository version it points to.

DCNRanker wraps a trained model so it drops into evaluation.
evaluate_ranker exactly like the LightGBM model: pass
feature_cols = DCN_DENSE_COLS + ["movie_idx"] so the frame slice carries
both the dense features and the embedding index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    """Listing 7.6: one DCN-v2 cross layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0, xl):
        return x0 * self.linear(xl) + xl


class DCNv2(nn.Module):
    """Listing 7.6: parallel cross network + deep MLP, single logit head."""

    def __init__(self, n_movies, dense_dim, emb_dim=32,
                 n_cross=3, mlp_dims=(128, 64)):
        super().__init__()
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        in_dim = emb_dim + dense_dim
        self.cross = nn.ModuleList(
            CrossLayer(in_dim) for _ in range(n_cross)
        )
        deep, d = [], in_dim
        for h in mlp_dims:
            deep += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.deep = nn.Sequential(*deep)
        self.head = nn.Linear(in_dim + d, 1)

    def forward(self, movie_ids, dense):
        x0 = torch.cat([self.movie_emb(movie_ids), dense], dim=1)
        xc = x0
        for layer in self.cross:
            xc = layer(x0, xc)
        xd = self.deep(x0)
        return self.head(torch.cat([xc, xd], dim=1)).squeeze(1)


def validate(model: nn.Module, loader, loss_fn) -> float:
    """Mean loss over a loader, in eval mode."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for movie, dense, y in loader:
            logits = model(movie, dense)
            total += loss_fn(logits, y).item() * len(y)
            n += len(y)
    return total / max(n, 1)


def train_dcn(model, loader, val_loader, epochs: int = 10, lr: float = 1e-3):
    """Pointwise training with early selection of the best epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state = float("inf"), None

    for _ in range(epochs):
        model.train()
        for movie, dense, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(movie, dense), y)
            loss.backward()
            opt.step()
        val = validate(model, val_loader, loss_fn)
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class DCNRanker:
    """sklearn-style predict() wrapper so DCN-v2 drops into evaluate_ranker.

    Expects the frame slice to contain dense_cols AND movie_idx, i.e.
    call evaluate_ranker with feature_cols = dense_cols + ["movie_idx"].
    Emits raw logits; the sigmoid is monotonic, so logits rank identically.
    """

    def __init__(self, model: DCNv2, scaler, dense_cols: list[str]):
        self.model = model
        self.scaler = scaler
        self.dense_cols = dense_cols

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        movie = torch.as_tensor(X["movie_idx"].to_numpy(copy=True))
        dense = torch.as_tensor(
            self.scaler.transform(X[self.dense_cols].to_numpy()),
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(movie, dense).numpy()
