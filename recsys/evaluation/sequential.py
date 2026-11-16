"""
recsys/evaluation/sequential.py
===============================
Next-item evaluation for the Chapter 6 sequential models (SASRec, BERT4Rec),
including the inference-time ablations of Listing 6.3.

Every history arrives left-padded with 0 (``truncate_and_pad``), so the most
recent item is always the rightmost position, whatever the history length.

  - ``history="full"``: the history as-is.
  - ``history="shuffled"`` (Ablation B): the real items permuted in place,
    padding left where it is. Tests whether the model uses order.
  - ``history="last_only"`` (Ablation C): only the most recent item. Tests
    whether a Markov-1 view of the history is almost as good.

Ablation A (no positional embeddings) changes the model, not the history: it
is a second model trained with ``use_pos_emb=False`` and evaluated with
``history="full"``.

Usage
-----
from recsys.evaluation.sequential import evaluate

evaluate(model, test_dataset)                       # full history
evaluate(model, test_dataset, history="shuffled")   # Ablation B
evaluate(model, test_dataset, history="last_only")  # Ablation C
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HISTORY_MODES = ("full", "shuffled", "last_only")


def shuffle_history(
  sequences: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
  """Permute each row's real (non-zero) items; padding stays on the left."""
  sequences = sequences.clone()
  for i in range(sequences.shape[0]):
    valid = sequences[i] != 0
    n_valid = int(valid.sum())
    if n_valid > 1:
      perm = torch.randperm(n_valid, generator=generator)
      sequences[i, valid] = sequences[i, valid][perm]
  return sequences


def last_item_only(sequences: torch.Tensor) -> torch.Tensor:
  """Keep only the most recent item: with left-padding, the rightmost position."""
  last = sequences[:, -1].clone()
  sequences = torch.zeros_like(sequences)
  sequences[:, -1] = last
  return sequences


@torch.no_grad()
def evaluate(
  model: torch.nn.Module,
  eval_dataset: Dataset,
  k: int = 10,
  batch_size: int = 256,
  device: torch.device | None = None,
  history: str = "full",
  seed: int = 42,
) -> dict[str, float]:
  """Full-catalog NDCG@k and HR@k on a held-out next-item target.

  ``eval_dataset`` yields ``(left-padded history, target)``. Scores are
  computed against the entire item embedding table, not a sampled subset:
  Krichene & Rendle (2020) show sampled evaluation ranks architectures
  inconsistently. The padding token is never recommended.

  ``history`` selects the full history or one of the inference-time
  ablations (see the module docstring); ``seed`` fixes the shuffle. A model
  with a ``mask_token`` (BERT4Rec) gets ``[MASK]`` appended after the
  ablation, and is read at that position. ``device`` defaults to the model's.
  """
  if history not in HISTORY_MODES:
    raise ValueError(f"history must be one of {HISTORY_MODES}, got {history!r}")
  if device is None:
    device = next(model.parameters()).device

  model.eval()
  loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
  rng = torch.Generator().manual_seed(seed)
  ndcgs, hits = [], []
  for seqs, targets in loader:
    if history == "shuffled":
      seqs = shuffle_history(seqs, rng)  #A
    elif history == "last_only":
      seqs = last_item_only(seqs)  #B
    seqs, targets = seqs.to(device), targets.to(device)

    if hasattr(model, "mask_token"):  #C
      mask_col = torch.full(
        (seqs.shape[0], 1), model.mask_token, dtype=torch.long, device=device
      )
      seqs = torch.cat([seqs, mask_col], dim=1)[:, -model.max_len :]

    user_state = model(seqs)[:, -1, :]  #D
    scores = user_state @ model.item_emb.weight[: model.num_items + 1].T  #E
    scores[:, 0] = -float("inf")

    topk = torch.topk(scores, k, dim=1).indices
    hit = topk == targets.unsqueeze(1)
    rank = hit.float().argmax(dim=1)
    has_hit = hit.any(dim=1)
    ndcg = torch.where(
      has_hit,
      1.0 / torch.log2(rank.float() + 2),
      torch.zeros_like(rank, dtype=torch.float),
    )
    ndcgs.extend(ndcg.cpu().tolist())
    hits.extend(has_hit.float().cpu().tolist())
  return {f"NDCG@{k}": float(np.mean(ndcgs)), f"HR@{k}": float(np.mean(hits))}


#A Ablation B: shuffle the real items in place
#B Ablation C: keep only the most recent item
#C BERT4Rec predicts at an appended [MASK]
#D The user state is the rightmost hidden vector
#E Score every item, without BERT4Rec's [MASK] row
