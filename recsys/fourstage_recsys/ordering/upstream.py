"""Chapter-5 upstream stages, packaged for the ordering stage (chapter 7).

The ordering model needs, for every user it trains or is tested on, the
candidate list chapter 5 would have produced and the cross-encoder score
of each candidate. This module fits the chapter-5 retrieval + scoring
models on a given training slice and exports those scored candidates.

It is called twice (see splits.py): once on the 0-70% slice to produce
the ranker's training rows, and once on the 0-80% slice (chapter 5's own
training set) to produce its test rows. Retrieval mirrors
notebooks/chapter-05/03_full_pipeline.ipynb: seed = the user's most recent
training item, ANN retrieval of `k_retrieve` candidates, history filtering.

The cross-encoder is chapter 7's own upstream scorer, trained here on every
call: user features are the mean of the user's history embeddings, training
rows pair each history item with banded retrieval negatives, and each row's
user vector leaves its own item out. Chapter 5's book model (time-split rows,
genre and year features; see scoring/cross_encoder_training.py) is a
different model and is not loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from recsys.data.preprocessing import user_item_lists
from recsys.fourstage_recsys.filtering.history_filtering import HistoryFiltering
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.retrieval.ann_retrieval import (
  ANNRetrieval,
  ANNRetrievalIndex,
)
from recsys.fourstage_recsys.retrieval.two_tower import (
  TwoTowerWithInfoNCE,
  create_positive_pairs,
  extract_embeddings,
  train_infonce,
)
from recsys.fourstage_recsys.scoring.cross_encoder import (
  CrossEncoderReranker,
  CrossEncoderScoring,
)

# Chapter-5 hyperparameters (01_similarity_learning.ipynb, 03_full_pipeline.ipynb)
CH5_INFONCE_PARAMS = dict(
  emb_dim=128,
  num_hard=4,
  num_rand=6,
  temperature=0.07,
  pool_start=200,
  pool_end=1000,
  epochs=20,
  warmup_epochs=2,
  batch_size=1024,
  lr=5e-3,
)
CH5_CROSS_ENCODER_EPOCHS = 5


# ---------------------------------------------------------------------------
# Upstream cross-encoder training (moved from scoring/cross_encoder.py, where
# chapter 5 now uses the time-split recipe in cross_encoder_training.py)
# ---------------------------------------------------------------------------


def build_cross_encoder_training(
  train_items_by_user: dict,
  neighbor_table: np.ndarray,
  num_items: int,
  num_hard: int = 2,
  num_random: int = 2,
  band: tuple = (0, None),
  seed: int = 42,
):
  """Training rows (user_id, item_idx, label, anchor) with mixed negatives.

  ``neighbor_table`` holds each item's retrieval neighbors in rank order
  (index space, shape (num_items, n_neighbors)). Hard negatives are drawn
  from ranks ``band[0]:band[1]`` of the positive's neighbor list. Starting
  the band past the very top avoids the false-negative trap from the
  hard-negative section: a user's closest unlabeled neighbors are exactly
  the movies they are most likely to watch next.

  ``anchor`` is the positive item each row was generated from. Training
  uses it to remove that item from the user's feature vector, so the
  model never sees a candidate inside its own user profile -- which is
  also how it will be served, after history filtering.
  """
  users, items, labels, anchors = [], [], [], []
  rng = np.random.default_rng(seed)
  pool = neighbor_table[:, band[0] : band[1]]
  n_neighbors = pool.shape[1]
  for user_id, pos_items in train_items_by_user.items():
    pos_set = set(pos_items)
    for item in pos_items:
      rows = [(item, 1.0)]
      for _ in range(num_hard):  # A
        neg = int(pool[item][rng.integers(n_neighbors)])
        tries = 0
        while neg in pos_set and tries < 10:
          neg = int(pool[item][rng.integers(n_neighbors)])
          tries += 1
        if neg in pos_set:
          neg = int(rng.integers(num_items))
        rows.append((neg, 0.0))
      for _ in range(num_random):  # B
        neg = int(rng.integers(num_items))
        while neg in pos_set:
          neg = int(rng.integers(num_items))
        rows.append((neg, 0.0))
      for it, lab in rows:
        users.append(user_id)
        items.append(it)
        labels.append(lab)
        anchors.append(item)
  return users, items, labels, anchors


# A In-distribution negatives: retrieval would surface them; the user did not choose
# B Random negatives keep the broad structure


def leave_one_out_user_features(
  user_features: dict,
  item_features: torch.Tensor,
  history_sizes: dict,
  users: list,
  anchors: list,
) -> torch.Tensor:
  """Per-row user vectors with the anchor item removed from the mean.

  user_features[u] is the mean of u's n history embeddings, so the mean
  without item a is (n * mean - e_a) / (n - 1). Without this, every
  positive row contains its own item inside the user vector and no
  negative does -- a shortcut the model learns instantly and that never
  exists at serving time, where candidates are filtered out of history.
  """
  means = torch.stack([user_features[u] for u in users])
  n = torch.tensor([float(history_sizes[u]) for u in users]).unsqueeze(1)
  loo = (n * means - item_features[anchors]) / (n - 1).clamp(min=1)
  return torch.where(n > 1, loo, means)


def train_cross_encoder(
  model: CrossEncoderReranker,
  user_features: dict,
  item_features: torch.Tensor,
  users: list,
  items: list,
  labels: list,
  anchors: list | None = None,
  history_sizes: dict | None = None,
  epochs: int = 5,
  batch_size: int = 4096,
  lr: float = 1e-3,
  device: str = "cpu",
  verbose: bool = True,
) -> CrossEncoderReranker:
  """BCE training over (user, item, label) rows.

  Pass ``anchors`` and ``history_sizes`` (from build_cross_encoder_training
  and the training histories) to train on leave-one-out user vectors --
  strongly recommended; see leave_one_out_user_features.
  """
  if anchors is not None:
    user_feat_matrix = leave_one_out_user_features(
      user_features, item_features, history_sizes, users, anchors
    )
  else:
    user_feat_matrix = torch.stack([user_features[u] for u in users])
  item_feat_matrix = item_features[items]
  label_tensor = torch.tensor(labels)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  bce = torch.nn.BCEWithLogitsLoss()
  model.to(device).train()
  n = len(labels)
  for epoch in range(epochs):
    perm = torch.randperm(n)
    total = 0.0
    for start in range(0, n, batch_size):
      idx = perm[start : start + batch_size]
      optimizer.zero_grad()
      logits = model(user_feat_matrix[idx].to(device), item_feat_matrix[idx].to(device))
      loss = bce(logits, label_tensor[idx].to(device))
      loss.backward()
      optimizer.step()
      total += loss.item() * len(idx)
    if verbose:
      print(f"epoch {epoch + 1}/{epochs} -- loss {total / n:.4f}")
  return model


@dataclass
class UpstreamModels:
  item_ids: list
  item_to_idx: dict
  query_embeddings: np.ndarray
  item_embeddings: np.ndarray
  cross_encoder: torch.nn.Module
  user_features: dict
  ann_index: ANNRetrievalIndex
  train_items_ids: dict = field(default_factory=dict)


def fit_upstream(
  train_pos: pd.DataFrame,
  item_ids: list,
  item_to_idx: dict,
  embeddings: tuple[np.ndarray, np.ndarray] | None = None,
  infonce_params: dict | None = None,
  cross_encoder_epochs: int = CH5_CROSS_ENCODER_EPOCHS,
  device: str = "cpu",
  seed: int = 42,
  verbose: bool = False,
) -> UpstreamModels:
  """Fit chapter-5 retrieval (two-tower InfoNCE) and scoring (cross-encoder).

  `train_pos` must carry userId, movieId, item_idx, timestamp. Pass
  `embeddings=(query, cand)` to reuse two-tower tables already trained
  on exactly this slice (e.g. chapter 5's embeddings.npz for the 80% cut).
  The cross-encoder is always trained here, on `train_pos`.
  """
  torch.manual_seed(seed)
  np.random.seed(seed)
  p = {**CH5_INFONCE_PARAMS, **(infonce_params or {})}
  num_items = len(item_ids)
  train_items_idx = user_item_lists(train_pos, item_col="item_idx")

  if embeddings is None:
    pairs = create_positive_pairs(train_items_idx, max_pairs_per_user=50)
    model = TwoTowerWithInfoNCE(
      num_items,
      emb_dim=p["emb_dim"],
      temperature=p["temperature"],
      num_hard=p["num_hard"],
      num_rand=p["num_rand"],
      pool_start=p["pool_start"],
      pool_end=p["pool_end"],
    )
    model, _ = train_infonce(
      model,
      pairs,
      epochs=p["epochs"],
      warmup_epochs=p["warmup_epochs"],
      batch_size=p["batch_size"],
      lr=p["lr"],
      device=device,
      verbose=verbose,
    )
    query, cand = extract_embeddings(model, device=device)
  else:
    query, cand = embeddings
  emb_dim = cand.shape[1]

  ann_index = ANNRetrievalIndex(index_type="hnsw", emb_dim=emb_dim)
  ann_index.add_items(item_ids, cand)

  item_features = torch.tensor(cand)
  user_features = {
    u: torch.tensor(cand[idxs].mean(axis=0)) for u, idxs in train_items_idx.items()
  }
  qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
  band = (p["pool_start"], p["pool_end"])  # same band as the two-tower
  _, neighbor_table = ann_index.index.search(
    np.ascontiguousarray(qn.astype(np.float32)), min(band[1], num_items)
  )
  users, items, labels, anchors = build_cross_encoder_training(
    train_items_idx, neighbor_table, num_items, band=band, seed=seed
  )
  history_sizes = {u: len(idxs) for u, idxs in train_items_idx.items()}
  cross_encoder = train_cross_encoder(
    CrossEncoderReranker(emb_dim, emb_dim),
    user_features,
    item_features,
    users,
    items,
    labels,
    anchors=anchors,
    history_sizes=history_sizes,
    epochs=cross_encoder_epochs,
    device=device,
    verbose=verbose,
  )
  return UpstreamModels(
    item_ids=item_ids,
    item_to_idx=item_to_idx,
    query_embeddings=query,
    item_embeddings=cand,
    cross_encoder=cross_encoder.eval(),
    user_features=user_features,
    ann_index=ann_index,
    train_items_ids=user_item_lists(train_pos, item_col="movieId"),
  )


def score_candidates(
  upstream: UpstreamModels,
  users,
  k_retrieve: int = 100,
  device: str = "cpu",
) -> pd.DataFrame:
  """Retrieve -> filter -> score for each user; one row per candidate.

  Returns userId, movieId, retrieval_similarity, cross_encoder_score.
  Users with no training history (no seed) are skipped.
  """
  retrieval = ANNRetrieval(
    upstream.ann_index, upstream.query_embeddings, upstream.item_to_idx
  )
  history = HistoryFiltering(upstream.train_items_ids)
  scorer = CrossEncoderScoring(
    upstream.cross_encoder,
    upstream.user_features,
    torch.tensor(upstream.item_embeddings),
    upstream.item_to_idx,
    device=device,
  )
  rows = []
  for user_id in users:
    seen = upstream.train_items_ids.get(user_id)
    if not seen:
      continue
    ctx = RecommendationContext(user_id=user_id, seed_items=[seen[-1]], k=k_retrieve)
    cands = retrieval.retrieve_similar_items(ctx.seed_items, k=k_retrieve)
    cands = scorer.score(history.filter(cands, ctx), ctx)
    rows.extend(
      (user_id, c.item_id, c.scores["similarity"], c.scores["cross_encoder"])
      for c in cands
    )
  return pd.DataFrame(
    rows, columns=["userId", "movieId", "retrieval_similarity", "cross_encoder_score"]
  )
