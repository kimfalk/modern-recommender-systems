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


class LowRankCrossLayer(nn.Module):
  """DCN-v2's low-rank cross layer: W = U V^T, so x0 * (U (V^T x_l) + b) + x_l."""

  def __init__(self, dim: int, rank: int):
    super().__init__()
    self.v = nn.Linear(dim, rank, bias=False)
    self.u = nn.Linear(rank, dim)

  def forward(self, x0, xl):
    return x0 * self.u(self.v(xl)) + xl


class DCNv2(nn.Module):
  """Listing 7.6: parallel cross network + deep MLP, single logit head.

  `cross_rank` switches to low-rank cross layers; `mlp_dropout` adds
  dropout after each hidden ReLU. Both default off, which is listing 7.6.
  """

  def __init__(self, n_movies, dense_dim, emb_dim=32, n_cross=3,
               mlp_dims=(128, 64), use_movie_emb=True, emb_dropout=0.0,
               mlp_dropout=0.0, cross_rank=None):
    super().__init__()
    self.use_movie_emb = use_movie_emb
    self.movie_emb = nn.Embedding(n_movies, emb_dim) if use_movie_emb else None
    # The embedding table holds most of the parameters and overfits first;
    # dropout on the looked-up vectors is the cheapest brake on it.
    self.emb_dropout = nn.Dropout(emb_dropout)
    in_dim = (emb_dim if use_movie_emb else 0) + dense_dim
    if cross_rank is None:
      self.cross = nn.ModuleList(CrossLayer(in_dim) for _ in range(n_cross))
    else:
      self.cross = nn.ModuleList(
        LowRankCrossLayer(in_dim, cross_rank) for _ in range(n_cross)
      )
    deep, d = [], in_dim
    for h in mlp_dims:
      deep += [nn.Linear(d, h), nn.ReLU()]
      if mlp_dropout > 0:
        deep.append(nn.Dropout(mlp_dropout))
      d = h
    self.deep = nn.Sequential(*deep)
    self.head = nn.Linear(in_dim + d, 1)

  # def forward(self, movie_ids, dense):
  #   x0 = torch.cat([self.movie_emb(movie_ids), dense], dim=1)
  #   xc = x0
  #   for layer in self.cross:
  #     xc = layer(x0, xc)
  #   xd = self.deep(x0)
  #   return self.head(torch.cat([xc, xd], dim=1)).squeeze(1)

  def forward(self, movie_ids, dense):
    if self.use_movie_emb:
      x0 = torch.cat([self.emb_dropout(self.movie_emb(movie_ids)), dense], dim=1)
    else:
      x0 = dense
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


def train_dcn(
  model,
  loader,
  val_loader=None,
  epochs: int = 10,
  lr: float = 1e-3,
  weight_decay: float = 0.0,
  emb_weight_decay: float | None = None,
  eval_fn=None,
  on_epoch=None,
  select_on: str | None = "val_loss",
):
  """Pointwise training; keeps the weights from the best epoch.

  Returns (model, history), where history holds one dict per epoch with
  train_loss and val_loss -- the curves that show under- or overfitting.
  `eval_fn(model) -> dict` adds metrics of your own per epoch (e.g.
  validation NDCG@10); `on_epoch(epoch, metrics)` is called after each
  epoch, which is where the notebook logs to MLflow.

  `select_on` names the metric that picks the best epoch: "val_loss" is
  minimized, any other key (e.g. "val_ndcg_at_10" from eval_fn) is
  maximized. BCE and NDCG disagree on which epoch is best, so select on
  the metric you report. None keeps the last epoch's weights, which is
  what a refit for a fixed number of epochs wants. Without a val_loader,
  val_loss is not computed.

  `emb_weight_decay` gives the movie embedding its own weight decay
  (None: same as `weight_decay`), so the table can be regularized hard
  without shrinking the cross and deep layers.
  """
  emb = [p for n, p in model.named_parameters() if n.startswith("movie_emb.")]
  rest = [p for n, p in model.named_parameters() if not n.startswith("movie_emb.")]
  groups = [{"params": rest, "weight_decay": weight_decay}]
  if emb:
    groups.append({"params": emb, "weight_decay":
                   weight_decay if emb_weight_decay is None else emb_weight_decay})
  opt = torch.optim.AdamW(groups, lr=lr)
  loss_fn = nn.BCEWithLogitsLoss()
  sign = 1.0 if select_on == "val_loss" else -1.0   # compare as "lower is better"
  best, best_state, history = float("inf"), None, []

  for epoch in range(epochs):
    model.train()
    total, n = 0.0, 0
    for movie, dense, y in loader:
      opt.zero_grad()
      loss = loss_fn(model(movie, dense), y)
      loss.backward()
      opt.step()
      total += loss.item() * len(y)
      n += len(y)
    metrics = {"train_loss": total / max(n, 1)}
    if val_loader is not None:
      metrics["val_loss"] = validate(model, val_loader, loss_fn)
    if eval_fn is not None:
      metrics.update(eval_fn(model))
    history.append({"epoch": epoch, **metrics})
    if on_epoch is not None:
      on_epoch(epoch, metrics)
    if select_on is not None and sign * metrics[select_on] < best:
      best = sign * metrics[select_on]
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  return model, history


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
