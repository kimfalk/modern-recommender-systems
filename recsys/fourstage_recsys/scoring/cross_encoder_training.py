"""
recsys/fourstage_recsys/scoring/cross_encoder_training.py
=========================================================
Training data and training loop for the MLP cross-encoder (Chapter 5,
Section 5.6), built so that time is respected inside the training period.

The scoring stage's null hypothesis is the retrieval order: a cross-encoder
earns its place only if rescoring the retrieved candidates puts more relevant
items at the top than the ANN score already does. To test that honestly, the
cross-encoder is trained on what it will see at serving time -- the candidates
retrieval returns for a user -- with labels from *later* than the features:

  history (training period, per user, in time order)
  |------- features -------|--- labels ---|--- validation labels ---|
  0                     label_start    val_start                   1.0

  - training rows: user features from [0, label_start), positives = items
    from [label_start, val_start) that retrieval returned, negatives = the
    other retrieved candidates;
  - validation rows: user features from [0, val_start), labels from
    [val_start, 1.0) -- used for early stopping only;
  - test: user features from the whole training period, labels from the test
    split. Nothing here reads the test split.

Features (both sides 148-d, so ``CrossEncoderReranker``'s ``u * i`` term
pairs like with like):

  - user: the retrieval query vector (query-tower embedding of the most recent
    item in the feature window), the share of the history in each genre, and
    the mean release year of the history;
  - item: the candidate-tower embedding, genre multi-hot, and release year.

The query-by-candidate product sums to the retrieval score, so the network
starts with the null ordering available as a feature.

Usage
-----
from recsys.fourstage_recsys.scoring.cross_encoder_training import (
  item_feature_matrix, user_feature_table, split_training_windows,
  retrieve_candidates, build_training_rows, fit_cross_encoder,
)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.ordering.features import GENRES

ITEM_GENRE_COLS = [f"item_is_{g}" for g in GENRES]


def _scale_year(years: np.ndarray) -> np.ndarray:
  """Release year on roughly the same scale as the other features."""
  return (years - 1900.0) / 100.0


def item_side_arrays(item_ids: list, genre_matrix: pd.DataFrame):
  """Per-item genre multi-hot (num_items, 19) and scaled release year.

  ``genre_matrix`` is ``ordering.features.build_genre_matrix(movies)``. Items
  without a parsable year get the catalog mean.
  """
  gm = genre_matrix.drop_duplicates("movieId").set_index("movieId")
  gm = gm.reindex([str(m) for m in item_ids])
  genres = gm[ITEM_GENRE_COLS].fillna(0).to_numpy(dtype=np.float32)
  years = gm["year"].to_numpy(dtype=np.float64)
  years = np.where(np.isnan(years), np.nanmean(years), years)
  return genres, _scale_year(years).astype(np.float32)


def item_feature_matrix(
  cand_embeddings: np.ndarray, item_genres: np.ndarray, item_years: np.ndarray
) -> torch.Tensor:
  """Item features in index order: [candidate embedding, genres, year]."""
  return torch.tensor(
    np.hstack([cand_embeddings, item_genres, item_years[:, None]]),
    dtype=torch.float32,
  )


def user_feature_table(
  histories: dict[str, list],
  query_embeddings: np.ndarray,
  item_genres: np.ndarray,
  item_years: np.ndarray,
) -> dict[str, torch.Tensor]:
  """User features from each user's history (item indices, oldest first).

  [query vector of the most recent item, genre shares, mean release year].
  Pass only the history that precedes the labels: this function cannot tell
  a feature window from a label window.
  """
  table = {}
  for user_id, hist in histories.items():
    if not hist:
      continue
    idx = np.asarray(hist)
    table[user_id] = torch.tensor(
      np.concatenate(
        [
          query_embeddings[idx[-1]],
          item_genres[idx].mean(axis=0),
          [item_years[idx].mean()],
        ]
      ),
      dtype=torch.float32,
    )
  return table


def split_training_windows(
  histories: dict[str, list], label_start: float = 0.70, val_start: float = 0.85
) -> dict:
  """Cut each user's training history into feature, label and validation windows.

  ``histories`` holds item indices in time order (``user_item_lists``).
  Returns ``{"train": {user: (features, labels)}, "val": {user: (features,
  labels)}}``, where validation features are everything before
  ``val_start``. Users whose history is too short for three non-empty
  windows are left out of both.
  """
  out = {"train": {}, "val": {}}
  for user_id, hist in histories.items():
    n = len(hist)
    a, b = int(n * label_start), int(n * val_start)
    if a < 1 or b <= a or n <= b:
      continue
    out["train"][user_id] = (list(hist[:a]), list(hist[a:b]))
    out["val"][user_id] = (list(hist[:b]), list(hist[b:]))
  return out


def retrieve_candidates(
  ann_index,
  query_embeddings: np.ndarray,
  histories: dict[str, list],
  item_ids: list,
  k: int = 200,
) -> dict[str, list[ScoredItem]]:
  """Top-k ANN candidates per user, in retrieval order.

  The query is the query-tower embedding of the user's most recent history
  item (the Table 5.1 protocol), and every history item is excluded from the
  pool, as ``HistoryFiltering`` would. ``ann_index`` is an
  ``ANNRetrievalIndex`` built over the candidate-tower embeddings with
  ``item_ids`` as its IDs.
  """
  out = {}
  for user_id, hist in histories.items():
    if not hist:
      continue
    seen = {item_ids[i] for i in hist}
    results = ann_index.retrieve(query_embeddings[hist[-1]], k=k, exclude=seen)
    out[user_id] = [
      ScoredItem(item_id=item_id, scores={"similarity": score})
      for item_id, score in results
    ]
  return out


def build_training_rows(
  candidates: dict[str, list[ScoredItem]],
  labels: dict[str, set],
  item_to_idx: dict,
  require_positive: bool = True,
):
  """Pointwise rows (user_id, item_idx, label) over each user's retrieved pool.

  Positives are the retrieved items in the user's label set; every other
  retrieved candidate is a negative. With ``require_positive``, users whose
  pool holds no label item are skipped: their list carries no ordering
  signal, only a tilt toward predicting zero.
  """
  users, items, ys = [], [], []
  for user_id, pool in candidates.items():
    positives = labels.get(user_id, set())
    pool_idx = [item_to_idx[c.item_id] for c in pool]
    flags = [1.0 if c.item_id in positives else 0.0 for c in pool]
    if require_positive and not any(flags):
      continue
    users.extend([user_id] * len(pool_idx))
    items.extend(pool_idx)
    ys.extend(flags)
  return users, np.asarray(items, dtype=np.int64), np.asarray(ys, dtype=np.float32)


def fit_cross_encoder(
  model: nn.Module,
  user_features: dict[str, torch.Tensor],
  item_features: torch.Tensor,
  users: list,
  items: np.ndarray,
  labels: np.ndarray,
  val_fn: Callable[[nn.Module], float] | None = None,
  max_epochs: int = 15,
  patience: int = 3,
  batch_size: int = 4096,
  lr: float = 1e-3,
  seed: int = 42,
  device: str = "cpu",
  verbose: bool = True,
):
  """BCE training with early stopping on ``val_fn`` (higher is better).

  ``val_fn(model) -> float`` is evaluated after every epoch -- typically
  validation NDCG@10 of the rescored pools, the metric being reported. The
  weights from the best epoch are kept. User feature rows are gathered per
  batch, so memory grows with the number of users, not rows.

  Returns ``(model, history)`` with one dict per epoch.
  """
  torch.manual_seed(seed)
  user_order = sorted(set(users))
  user_pos = {u: i for i, u in enumerate(user_order)}
  user_matrix = torch.stack([user_features[u] for u in user_order])
  user_rows = torch.tensor([user_pos[u] for u in users])
  item_rows = torch.as_tensor(items)
  targets = torch.as_tensor(labels)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  bce = nn.BCEWithLogitsLoss()
  model.to(device)
  n = len(targets)
  best, best_state, since_best, history = -np.inf, None, 0, []
  for epoch in range(max_epochs):
    t0 = time.perf_counter()
    model.train()
    perm = torch.randperm(n)
    total = 0.0
    for start in range(0, n, batch_size):
      idx = perm[start : start + batch_size]
      optimizer.zero_grad()
      logits = model(
        user_matrix[user_rows[idx]].to(device), item_features[item_rows[idx]].to(device)
      )
      loss = bce(logits, targets[idx].to(device))
      loss.backward()
      optimizer.step()
      total += loss.item() * len(idx)
    row = {"epoch": epoch + 1, "train_loss": total / n}
    if val_fn is not None:
      model.eval()
      row["val_metric"] = float(val_fn(model))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    history.append(row)
    if verbose:
      print(
        ", ".join(
          f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()
        )
      )
    if val_fn is None:
      continue
    if row["val_metric"] > best:
      best, since_best = row["val_metric"], 0
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
      since_best += 1
      if since_best >= patience:
        break
  if best_state is not None:
    model.load_state_dict(best_state)
  model.eval()
  return model, history
