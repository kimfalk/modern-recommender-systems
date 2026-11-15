"""Tests for the Chapter 6 next-item evaluation and its ablations (Listing 6.3).

Small synthetic tensors only: no network, no data files, no training.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.evaluation.sequential import (
  evaluate,
  last_item_only,
  shuffle_history,
)
from recsys.fourstage_recsys.retrieval.sequential import (
  BERT4Rec,
  SASRec,
  SequentialEvalDataset,
)

NUM_ITEMS = 20
# Left-padded, as truncate_and_pad produces: the newest item is rightmost.
LEFT_PADDED = torch.tensor([[0, 0, 5, 7], [1, 2, 3, 4], [0, 0, 0, 9]])


class LastPositionEcho(nn.Module):
  """Hidden state = one-hot of the item at each position, so the user state
  is the rightmost item and the top recommendation is that item itself."""

  def __init__(self, num_items: int = NUM_ITEMS, max_len: int = 4):
    super().__init__()
    self.num_items, self.max_len = num_items, max_len
    self.item_emb = nn.Embedding(num_items + 1, num_items + 1)
    with torch.no_grad():
      self.item_emb.weight.copy_(torch.eye(num_items + 1))

  def forward(self, sequences):
    return self.item_emb(sequences)


def _dataset(histories, targets, max_len=4):
  return SequentialEvalDataset(histories, targets, max_len=max_len)


def test_last_item_only_keeps_the_rightmost_item_of_left_padded_rows():
  out = last_item_only(LEFT_PADDED)
  assert out.tolist() == [[0, 0, 0, 7], [0, 0, 0, 4], [0, 0, 0, 9]]
  assert LEFT_PADDED[0].tolist() == [0, 0, 5, 7]  # input untouched


def test_shuffle_history_permutes_items_and_leaves_padding_in_place():
  out = shuffle_history(LEFT_PADDED, torch.Generator().manual_seed(0))
  assert ((out == 0) == (LEFT_PADDED == 0)).all()
  for row_in, row_out in zip(LEFT_PADDED, out):
    assert sorted(row_in.tolist()) == sorted(row_out.tolist())


def test_shuffle_history_is_deterministic_in_the_generator_seed():
  a = shuffle_history(LEFT_PADDED, torch.Generator().manual_seed(3))
  b = shuffle_history(LEFT_PADDED, torch.Generator().manual_seed(3))
  assert torch.equal(a, b)


def test_evaluate_full_and_last_only_hit_the_newest_item():
  # Short histories, so most of each padded row is padding.
  data = _dataset([[5, 7], [1, 2, 3, 4], [9]], [7, 4, 9])
  model = LastPositionEcho()
  for history in ("full", "last_only"):
    metrics = evaluate(model, data, k=1, history=history)
    assert metrics == {"NDCG@1": 1.0, "HR@1": 1.0}, history


def test_evaluate_names_metrics_by_k_and_misses_when_target_is_not_newest():
  # Each target is an older history item, so the echo model ranks it below the newest.
  data = _dataset([[3, 5], [4, 6]], [3, 4])
  metrics = evaluate(LastPositionEcho(), data, k=1)
  assert metrics == {"NDCG@1": 0.0, "HR@1": 0.0}


def test_evaluate_shuffled_is_seeded():
  data = _dataset([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]], [4, 8, 11])
  model = LastPositionEcho()
  a = evaluate(model, data, k=1, history="shuffled", seed=1)
  b = evaluate(model, data, k=1, history="shuffled", seed=1)
  assert a == b


def test_evaluate_rejects_an_unknown_history_mode():
  with pytest.raises(ValueError):
    evaluate(LastPositionEcho(), _dataset([[1]], [1]), history="reversed")


@pytest.mark.parametrize("model_cls", [SASRec, BERT4Rec])
@pytest.mark.parametrize("history", ["full", "shuffled", "last_only"])
def test_evaluate_runs_on_the_package_models(model_cls, history):
  torch.manual_seed(0)
  model = model_cls(
    num_items=NUM_ITEMS, max_len=6, hidden_dim=8, num_layers=1, num_heads=2
  )
  data = _dataset([[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]], [4, 6, 11], max_len=6)
  metrics = evaluate(model, data, k=10, history=history)
  assert 0.0 <= metrics["NDCG@10"] <= metrics["HR@10"] <= 1.0
