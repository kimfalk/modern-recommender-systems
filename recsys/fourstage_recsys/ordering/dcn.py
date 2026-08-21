"""DCN-v2 ranker in PyTorch (chapter 7, listings 7.6-7.9).

The model consumes exactly the feature table produced by
``make_ranking_dataset`` -- the same rows LightGBM trains on -- plus one
thing trees cannot digest: the raw movieId, as an embedding.  That single
addition is the "high-cardinality" advantage discussed in section 7.4.2.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import per_group_metrics
from .features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Listing 7.6 -- embedding layer for heterogeneous features
# ---------------------------------------------------------------------------
class FeatureEmbedder(nn.Module):
    """Turn categorical indices + dense features into one flat vector x0."""

    def __init__(
        self,
        cat_cardinalities: dict[str, int],
        n_dense: int,
        emb_dim: int = 32,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                # Index 0 is reserved for unknown/cold items -- the padding
                # trick from chapter 6.
                name: nn.Embedding(card + 1, emb_dim, padding_idx=0)
                for name, card in cat_cardinalities.items()
            }
        )
        self.dense_norm = nn.BatchNorm1d(n_dense)
        self.out_dim = emb_dim * len(cat_cardinalities) + n_dense

    def forward(
        self, cats: dict[str, torch.Tensor], dense: torch.Tensor
    ) -> torch.Tensor:
        embs = [self.embeddings[name](idx) for name, idx in cats.items()]
        return torch.cat(embs + [self.dense_norm(dense)], dim=-1)


# ---------------------------------------------------------------------------
# Listing 7.7 -- the low-rank cross network
# ---------------------------------------------------------------------------
class CrossLayerV2(nn.Module):
    """One DCN-v2 cross layer: x_{l+1} = x0 * (W x_l + b) + x_l, W ~ U V^T."""

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.U = nn.Linear(rank, dim, bias=False)
        self.V = nn.Linear(dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * (self.U(self.V(xl)) + self.bias) + xl


class CrossNetworkV2(nn.Module):
    """A stack of cross layers; each adds one degree of feature interaction."""

    def __init__(self, dim: int, n_layers: int = 3, rank: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossLayerV2(dim, rank) for _ in range(n_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


# ---------------------------------------------------------------------------
# Listing 7.8 -- full model assembly
# ---------------------------------------------------------------------------
class DCNv2Ranker(nn.Module):
    """Cross network + deep network in parallel, one linear scoring head."""

    def __init__(
        self,
        embedder: FeatureEmbedder,
        n_cross_layers: int = 3,
        rank: int = 64,
        deep_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedder = embedder
        d = embedder.out_dim
        self.cross = CrossNetworkV2(d, n_cross_layers, rank)

        deep: list[nn.Module] = []
        prev = d
        for h in deep_dims:
            deep += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep)

        self.head = nn.Linear(d + deep_dims[-1], 1)

    def forward(
        self, cats: dict[str, torch.Tensor], dense: torch.Tensor
    ) -> torch.Tensor:
        x0 = self.embedder(cats, dense)
        x_cross = self.cross(x0)
        x_deep = self.deep(x0)
        return self.head(torch.cat([x_cross, x_deep], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Listing 7.9 -- sampled-softmax listwise loss and training loop
# ---------------------------------------------------------------------------
def listwise_loss(pos_score: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Sampled softmax over [positive | negatives]; correct class is column 0."""
    logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def bpr_loss(pos_score: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """The pairwise alternative, one line, for the 7.2.1 experiment."""
    return -F.logsigmoid(pos_score.unsqueeze(1) - neg_scores).mean()


class MovieIndex:
    """Map raw movieIds to contiguous embedding indices; unknown -> 0."""

    def __init__(self, train_movie_ids: np.ndarray):
        self.to_idx = {int(m): i + 1 for i, m in enumerate(np.unique(train_movie_ids))}
        self.cardinality = len(self.to_idx)

    def __call__(self, movie_ids: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.to_idx.get(int(m), 0) for m in movie_ids], dtype=np.int64
        )


def frame_to_tensors(
    frame: pd.DataFrame,
    movie_index: MovieIndex,
    feature_cols: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorize an assembled example frame -> (movie_idx, dense)."""
    cols = feature_cols or FEATURE_COLS
    movie_idx = torch.from_numpy(movie_index(frame["movieId"].to_numpy()))
    dense = torch.from_numpy(
        frame[cols].to_numpy(dtype=np.float32, na_value=0.0)
    )
    return movie_idx, dense


class GroupedRankingDataset(Dataset):
    """One example = one positive plus K sampled negatives from its group.

    Batches built from this dataset keep a positive and its user's
    negatives together, which is what the listwise loss needs.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        groups: np.ndarray,
        movie_index: MovieIndex,
        n_negatives: int = 8,
        feature_cols: list[str] | None = None,
        seed: int = 42,
    ):
        self.movie_idx, self.dense = frame_to_tensors(
            frame, movie_index, feature_cols
        )
        rng = np.random.default_rng(seed)
        labels = frame["label"].to_numpy()

        self.examples: list[tuple[int, np.ndarray]] = []
        start = 0
        for size in groups:
            rows = np.arange(start, start + int(size))
            pos_rows = rows[labels[rows] > 0]
            neg_rows = rows[labels[rows] == 0]
            if len(neg_rows) > 0:
                for p in pos_rows:
                    negs = rng.choice(
                        neg_rows,
                        size=n_negatives,
                        replace=len(neg_rows) < n_negatives,
                    )
                    self.examples.append((int(p), negs))
            start += int(size)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        p, negs = self.examples[i]
        rows = np.concatenate([[p], negs])
        return self.movie_idx[rows], self.dense[rows]


def train_epoch(
    model: DCNv2Ranker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn=listwise_loss,
) -> float:
    """One pass over the grouped training data (listing 7.9)."""
    model.train()
    total = 0.0
    for movie_idx, dense in loader:
        b, slots, d = dense.shape  # slots = 1 positive + K negatives
        movie_idx = movie_idx.to(device).reshape(b * slots)
        dense = dense.to(device).reshape(b * slots, d)

        scores = model({"movie": movie_idx}, dense).reshape(b, slots)
        loss = loss_fn(scores[:, 0], scores[:, 1:])

        optimizer.zero_grad()
        loss.backward()
        # Chapter 6's hard-won lesson about exploding gradients, applied
        # preemptively.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def predict_scores(
    model: DCNv2Ranker,
    frame: pd.DataFrame,
    movie_index: MovieIndex,
    device: torch.device,
    feature_cols: list[str] | None = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """Score a flat example frame -- same contract as ``Booster.predict``."""
    model.eval()
    movie_idx, dense = frame_to_tensors(frame, movie_index, feature_cols)
    out = []
    for start in range(0, len(dense), batch_size):
        sl = slice(start, start + batch_size)
        scores = model(
            {"movie": movie_idx[sl].to(device)}, dense[sl].to(device)
        )
        out.append(scores.cpu().numpy())
    return np.concatenate(out)


def fit_dcn(
    model: DCNv2Ranker,
    train_ds: GroupedRankingDataset,
    valid_frame: pd.DataFrame,
    y_valid: np.ndarray,
    groups_valid: np.ndarray,
    movie_index: MovieIndex,
    device: torch.device,
    feature_cols: list[str] | None = None,
    epochs: int = 20,
    patience: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_fn=listwise_loss,
    verbose: bool = True,
) -> dict:
    """Train with early stopping on validation NDCG@10 -- the same stopping
    rule as the LightGBM ranker in listing 7.3, which keeps the head-to-head
    comparison fair."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best = {"ndcg@10": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, loss_fn=loss_fn)
        scores = predict_scores(model, valid_frame, movie_index, device, feature_cols)
        metrics = per_group_metrics(scores, y_valid, groups_valid, ks=(10,))
        history.append({"epoch": epoch, "loss": loss, **metrics})
        if verbose:
            print(
                f"epoch {epoch:02d}  loss {loss:.4f}  "
                f"valid ndcg@10 {metrics['ndcg@10']:.4f}  mrr {metrics['mrr']:.4f}"
            )
        if metrics["ndcg@10"] > best["ndcg@10"]:
            best = {
                "ndcg@10": metrics["ndcg@10"],
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }
        elif epoch - best["epoch"] >= patience:
            if verbose:
                print(f"early stopping: no improvement since epoch {best['epoch']}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_ndcg@10": best["ndcg@10"], "history": history}
