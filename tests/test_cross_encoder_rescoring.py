"""Tests for the cross-encoder rescoring experiment (chapter 5, section 5.6).

Small synthetic tensors only: no network, no data files, no training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys.evaluation.rescoring import (  # noqa: E402
  NULL_ROW,
  POPULARITY_ROW,
  RESCORED_ROW,
  evaluate_orderings,
  evaluate_pipelines,
  paired_bootstrap,
  validation_metric,
)
from recsys.fourstage_recsys.filtering.history_filtering import (  # noqa: E402
  HistoryFiltering,
)
from recsys.fourstage_recsys.item_context import ScoredItem  # noqa: E402
from recsys.fourstage_recsys.ordering.features import (  # noqa: E402
  GENRES,
  build_genre_matrix,
)
from recsys.fourstage_recsys.ordering.weighted_ranker import (  # noqa: E402
  WeightedRanker,
)
from recsys.fourstage_recsys.pipeline import FourStageRecommender  # noqa: E402
from recsys.fourstage_recsys.recsys_context import (  # noqa: E402
  RecommendationContext,
)
from recsys.fourstage_recsys.retrieval.ann_retrieval import (  # noqa: E402
  ANNRetrievalIndex,
)
from recsys.fourstage_recsys.scoring.cross_encoder import (  # noqa: E402
  CrossEncoderReranker,
)
from recsys.fourstage_recsys.scoring.cross_encoder_training import (  # noqa: E402
  build_training_rows,
  item_feature_matrix,
  item_side_arrays,
  retrieve_candidates,
  split_training_windows,
  user_feature_table,
)
from recsys.fourstage_recsys.scoring.passthrough import (  # noqa: E402
  PassThroughScoring,
)
from recsys.fourstage_recsys.scoring.popularity import (  # noqa: E402
  PopularityScoring,
)

DIM = 8


@pytest.fixture
def catalog():
  rng = np.random.default_rng(0)
  n = 30
  item_ids = [str(100 + i) for i in range(n)]
  movies = pd.DataFrame(
    {
      "movieId": item_ids,
      "title": [f"Movie {i} ({1950 + i})" for i in range(n - 1)] + ["No Year"],
      "genres": ["Action|Comedy" if i % 2 else "Drama" for i in range(n)],
    }
  )
  query = rng.normal(size=(n, DIM)).astype(np.float32)
  cand = rng.normal(size=(n, DIM)).astype(np.float32)
  query /= np.linalg.norm(query, axis=1, keepdims=True)
  cand /= np.linalg.norm(cand, axis=1, keepdims=True)
  return item_ids, movies, query, cand


class FixedScorer:
  """A pipeline-style scorer that writes a precomputed score per item."""

  def __init__(self, scores):
    self.scores = scores

  def score(self, candidates, context):
    for c in candidates:
      c.scores["cross_encoder"] = self.scores[c.item_id]
    return candidates


def pool(ids, sims=None):
  sims = sims if sims is not None else np.linspace(1, 0, len(ids))
  return [
    ScoredItem(item_id=i, scores={"similarity": float(s)})
    for i, s in zip(ids, sims, strict=True)
  ]


# --- time windows -------------------------------------------------------


def test_windows_are_contiguous_and_in_time_order():
  hist = {"u": list(range(20)), "short": [1, 2]}
  w = split_training_windows(hist, label_start=0.7, val_start=0.85)
  feats, labels = w["train"]["u"]
  vfeats, vlabels = w["val"]["u"]
  assert feats == list(range(14)) and labels == [14, 15, 16]
  assert vfeats == list(range(17)) and vlabels == [17, 18, 19]
  assert max(feats) < min(labels) and max(vfeats) < min(vlabels)  # labels later
  assert "short" not in w["train"] and "short" not in w["val"]


# --- features -----------------------------------------------------------


def test_item_side_arrays_fill_missing_year(catalog):
  item_ids, movies, _, _ = catalog
  genres, years = item_side_arrays(item_ids, build_genre_matrix(movies))
  assert genres.shape == (len(item_ids), len(GENRES))
  assert genres[0, GENRES.index("Drama")] == 1.0
  assert genres[1, GENRES.index("Comedy")] == 1.0
  assert np.isfinite(years).all()
  assert years[-1] == pytest.approx(years[:-1].mean())  # "No Year" -> mean


def test_user_and_item_features_match_reranker(catalog):
  item_ids, movies, query, cand = catalog
  genres, years = item_side_arrays(item_ids, build_genre_matrix(movies))
  items = item_feature_matrix(cand, genres, years)
  users = user_feature_table({"u": [3, 5, 7]}, query, genres, years)
  assert users["u"].shape[0] == items.shape[1] == DIM + len(GENRES) + 1
  assert torch.allclose(users["u"][:DIM], torch.tensor(query[7]))  # last item
  assert torch.allclose(
    users["u"][DIM:-1], torch.tensor(genres[[3, 5, 7]].mean(axis=0))
  )
  model = CrossEncoderReranker(items.shape[1], items.shape[1])  # untrained
  logits = model(users["u"].expand(4, -1), items[:4])
  assert logits.shape == (4,)


# --- candidates and rows ------------------------------------------------


def test_retrieve_candidates_excludes_history_and_is_sorted(catalog):
  item_ids, _, query, cand = catalog
  index = ANNRetrievalIndex(index_type="flat", emb_dim=DIM)
  index.add_items(item_ids, cand)
  out = retrieve_candidates(index, query, {"u": [0, 1, 2]}, item_ids, k=10)
  got = [c.item_id for c in out["u"]]
  assert len(got) == 10
  assert not {"100", "101", "102"} & set(got)
  sims = [c.scores["similarity"] for c in out["u"]]
  assert sims == sorted(sims, reverse=True)
  expected = cand @ query[2]  # query = most recent history item
  expected[[0, 1, 2]] = -np.inf
  assert got == [item_ids[i] for i in np.argsort(-expected)[:10]]


def test_training_rows_label_only_retrieved_label_items():
  item_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
  cands = {"u1": pool(["a", "b", "c"]), "u2": pool(["c", "d"])}
  labels = {"u1": {"b", "d"}, "u2": {"a"}}  # u2's label item was not retrieved
  users, items, ys = build_training_rows(cands, labels, item_to_idx)
  assert users == ["u1", "u1", "u1"]
  assert items.tolist() == [0, 1, 2] and ys.tolist() == [0.0, 1.0, 0.0]
  users, _, ys = build_training_rows(cands, labels, item_to_idx, require_positive=False)
  assert len(users) == 5 and ys.sum() == 1.0


# --- evaluation ---------------------------------------------------------


def test_perfect_rescoring_and_pool_recall_unchanged():
  ids = [f"i{j}" for j in range(20)]
  relevant = {"i15", "i19"}  # retrieval ranks them 16th and 20th
  scores = {i: (1.0 if i in relevant else 0.0) for i in ids}
  summary, per_user = evaluate_orderings(
    {"u": pool(ids)},
    {"u": relevant},
    {RESCORED_ROW: (FixedScorer(scores), "cross_encoder")},
    k_top=10,
    k_recall=10,
  )
  assert summary.loc[NULL_ROW, "NDCG@10"] == 0.0
  assert summary.loc[RESCORED_ROW, "NDCG@10"] == pytest.approx(1.0)
  assert summary.loc[RESCORED_ROW, "Precision@10"] == pytest.approx(0.2)
  # Rescoring cannot change the pool ...
  assert (
    summary.loc[NULL_ROW, "Recall@pool"] == summary.loc[RESCORED_ROW, "Recall@pool"]
  )
  # ... but it decides what survives a cut shorter than the pool.
  assert summary.loc[NULL_ROW, "Recall@10"] == 0.0
  assert summary.loc[RESCORED_ROW, "Recall@10"] == 1.0
  assert summary["users"].tolist() == [1, 1]
  assert per_user.shape[0] == 1


def test_same_users_for_both_orderings_and_ties_keep_retrieval_order():
  ids = ["a", "b", "c"]
  cands = {"u1": pool(ids), "u2": pool(ids), "u3": []}
  relevance = {"u1": {"a"}, "u2": set(), "u3": {"a"}}  # only u1 is evaluable
  summary, _ = evaluate_orderings(
    cands,
    relevance,
    {RESCORED_ROW: (FixedScorer(dict.fromkeys(ids, 0.0)), "cross_encoder")},
    k_top=1,
    k_recall=1,
  )
  assert summary["users"].tolist() == [1, 1]
  # All scores tie, so the rescored order is the retrieval order.
  assert summary.loc[RESCORED_ROW, "NDCG@1"] == summary.loc[NULL_ROW, "NDCG@1"] == 1.0


def test_validation_metric_runs_through_the_scorer():
  ids = ["a", "b", "c"]
  val_fn = validation_metric(
    {"u": pool(ids)},
    {"u": {"c"}},
    make_scorer=lambda model: FixedScorer({"a": 0.0, "b": 0.0, "c": model}),
  )
  assert val_fn(1.0) == pytest.approx(1.0)  # c rescored to the top
  assert val_fn(-1.0) == pytest.approx(0.5)  # c stays third: 1/log2(4)


def test_popularity_row_uses_training_counts_and_keeps_ties_in_retrieval_order():
  ids = ["a", "b", "c", "d"]
  train = pd.DataFrame({"movieId": ["d", "d", "d", "b", "c"]})  # a unseen in training
  summary, per_user = evaluate_orderings(
    {"u": pool(ids)},
    {"u": {"d"}},
    {POPULARITY_ROW: (PopularityScoring(train), "popularity")},
    k_top=1,
    k_recall=1,
  )
  assert list(summary.index) == [NULL_ROW, POPULARITY_ROW]
  assert summary.loc[NULL_ROW, "NDCG@1"] == 0.0
  assert summary.loc[POPULARITY_ROW, "NDCG@1"] == 1.0  # d is most popular
  # b and c tie on one interaction each: popularity order keeps b before c.
  scored = PopularityScoring(train).score(pool(ids), None)
  assert [c.item_id for c in sorted(scored, key=lambda c: -c.scores["popularity"])] == [
    "d",
    "b",
    "c",
    "a",
  ]


def test_paired_bootstrap_counts_and_interval():
  per_user = pd.DataFrame(
    {
      ("x", "NDCG@10"): [0.0, 0.0, 0.5, 1.0],
      ("y", "NDCG@10"): [1.0, 0.0, 0.5, 0.0],
    }
  )
  out = paired_bootstrap(per_user, "x", "y", n_boot=200)
  assert out["diff"] == pytest.approx(0.0)
  assert (out["better"], out["worse"], out["same"]) == (1, 1, 2)
  assert out["ci_low"] <= out["diff"] <= out["ci_high"]


# --- the pipeline ---------------------------------------------------------


class ListRetrieval:
  """Retrieval stub: a fixed ranked list, recording the k it was asked for."""

  num_items = 6

  def __init__(self, ids):
    self.ids, self.asked = ids, []

  def retrieve_similar_items(self, seed_ids, k=100):
    self.asked.append(k)
    return pool(self.ids[:k])


def test_retrieval_k_defaults_to_100_and_is_configurable():
  ids = [f"i{j}" for j in range(6)]
  args = (HistoryFiltering({}), PassThroughScoring(), WeightedRanker({"similarity": 1}))
  default = FourStageRecommender(ListRetrieval(ids), *args)
  wide = FourStageRecommender(ListRetrieval(ids), *args, retrieval_k=4)
  ctx = RecommendationContext(user_id="u", seed_items=["s"], k=10)
  assert len(default.recommend(ctx)) == 6 and default.retrieval.asked == [100]
  assert [c.item_id for c in wide.recommend(ctx)] == ids[:4]
  assert wide.retrieval.asked == [4]


def test_evaluate_pipelines_filters_history_before_scoring():
  ids = ["h", "a", "b", "c"]  # "h" is in the user's history
  history = HistoryFiltering({"u": ["h"]})
  recs = {
    NULL_ROW: FourStageRecommender(
      ListRetrieval(ids),
      history,
      PassThroughScoring(),
      WeightedRanker({"similarity": 1}),
      retrieval_k=4,
    ),
    RESCORED_ROW: FourStageRecommender(
      ListRetrieval(ids),
      history,
      FixedScorer({"h": 9.0, "a": 0.0, "b": 0.0, "c": 1.0}),
      WeightedRanker({"cross_encoder": 1}),
      retrieval_k=4,
    ),
  }
  summary, _ = evaluate_pipelines(
    recs, {"u": ["s"]}, {"u": {"c"}}, pool_k=4, k_top=1, k_recall=1
  )
  assert summary.attrs["mean_pool_size"] == 3  # 4 retrieved, history removed
  assert summary.loc[NULL_ROW, "NDCG@1"] == 0.0
  assert summary.loc[RESCORED_ROW, "NDCG@1"] == 1.0
  assert (
    summary.loc[NULL_ROW, "Recall@pool"] == summary.loc[RESCORED_ROW, "Recall@pool"]
  )
