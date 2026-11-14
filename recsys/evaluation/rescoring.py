"""
recsys/evaluation/rescoring.py
==============================
Does the scoring stage beat retrieval order? (Chapter 5, Section 5.6)

Retrieval order is the null hypothesis for the scoring stage: the candidates
already arrive sorted by ANN similarity, and a scorer is only worth its
latency if re-sorting the *same* pool puts more relevant items on top. A
popularity ordering of the same pool is the cheap alternative it also has to
beat: a learned scorer that mostly learned popularity is not worth serving.

Two harnesses, the same metrics (the Chapter 4 functions):

  - ``evaluate_orderings`` re-sorts precomputed pools directly -- the pools
    the cross-encoder trains and validates on;
  - ``evaluate_pipelines`` runs complete ``FourStageRecommender`` instances,
    so retrieval, filtering, scoring and ordering are the serving code.

Re-sorting cannot change which items are in the pool, so recall over the
whole pool (``Recall@pool``) is identical across orderings by construction.
Recall at a cut shorter than the pool (e.g. Recall@100 of a 200-item pool)
*can* change: the ordering decides what survives the cut.

Usage
-----
from recsys.evaluation.rescoring import (
  evaluate_orderings, evaluate_pipelines, paired_bootstrap, validation_metric,
)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext

NULL_ROW = "Retrieval order (null)"
POPULARITY_ROW = "Popularity order"
RESCORED_ROW = "Cross-encoder rescoring"


def _metrics(ranked: list, relevant: set, k_top: int, k_recall: int, pool: int):
  return {
    f"Precision@{k_top}": precision_at_k(ranked, relevant, k_top),
    f"NDCG@{k_top}": ndcg_at_k(ranked, relevant, k_top),
    f"Recall@{k_recall}": recall_at_k(ranked, relevant, k_recall),
    "Recall@pool": recall_at_k(ranked, relevant, pool),
  }


def _tables(rows: dict, order: list, pool_sizes: list):
  """Per-user and summary DataFrames from {user: {ordering: metrics}}."""
  if not rows:
    return pd.DataFrame(), pd.DataFrame()
  metric_names = list(next(iter(rows.values()))[order[0]])
  per_user = pd.DataFrame(
    {(o, m): {u: r[o][m] for u, r in rows.items()} for o in order for m in metric_names}
  )
  summary = per_user.mean().unstack(level=1).loc[order, metric_names]
  summary["users"] = len(rows)
  summary.attrs["mean_pool_size"] = float(np.mean(pool_sizes))
  return summary, per_user


def evaluate_orderings(
  candidates: dict[str, list[ScoredItem]],
  relevance: dict[str, set],
  scorers: dict,
  k_top: int = 10,
  k_recall: int = 100,
  retrieval_key: str = "similarity",
):
  """Retrieval order vs. one or more rescored orders of the same pools.

  ``candidates`` holds each user's pool in retrieval order (the
  ``retrieval_key`` score descending). ``scorers`` maps a row name to a
  ``(scorer, score_key)`` pair: any pipeline ``Scorer`` (e.g.
  ``CrossEncoderScoring``, ``PopularityScoring``) and the score it writes onto
  the items. Users without relevant items, without candidates, or without
  features for some scorer are skipped -- every row covers the same users.
  Ties keep retrieval order.

  Returns ``(summary, per_user)``: one row per ordering (``NULL_ROW`` first)
  of mean metrics plus a ``users`` column, and a per-user DataFrame
  (MultiIndex columns: ordering, metric) for paired comparisons.
  """
  order = [NULL_ROW, *scorers]
  rows, pool_sizes = {}, []
  for user_id, pool in candidates.items():
    relevant = relevance.get(user_id)
    if not relevant or not pool:
      continue
    by_retrieval = sorted(pool, key=lambda c: -c.scores[retrieval_key])
    ctx = RecommendationContext(user_id=user_id, seed_items=[], k=k_top)
    ranked = {NULL_ROW: [c.item_id for c in by_retrieval]}
    for name, (scorer, key) in scorers.items():
      scored = scorer.score(list(by_retrieval), ctx)
      if any(key not in c.scores for c in scored):
        break  # the scorer had no features for this user
      ranked[name] = [c.item_id for c in sorted(scored, key=lambda c: -c.scores[key])]
    else:
      pool_sizes.append(len(pool))
      rows[user_id] = {
        name: _metrics(ids, relevant, k_top, k_recall, len(pool))
        for name, ids in ranked.items()
      }
  return _tables(rows, order, pool_sizes)


def evaluate_pipelines(
  recommenders: dict,
  seeds: dict[str, list],
  relevance: dict[str, set],
  pool_k: int,
  k_top: int = 10,
  k_recall: int = 100,
):
  """The same metrics, computed on what complete pipelines return.

  ``recommenders`` maps a row name to a ``FourStageRecommender``; each user
  is served once per pipeline with ``seeds[user]`` as the seed items and
  ``k=pool_k``, so the full ordered pool comes back and recall can be cut at
  any depth. ``Recall@pool`` is recall over whatever the pipeline returned
  (retrieval ``retrieval_k`` minus what filtering removed). Users without
  relevant items, without a seed, or with an empty result are skipped.
  """
  order = list(recommenders)
  rows, pool_sizes = {}, []
  for user_id, relevant in relevance.items():
    if not relevant or not seeds.get(user_id):
      continue
    ctx = RecommendationContext(user_id=user_id, seed_items=seeds[user_id], k=pool_k)
    ranked = {
      name: [c.item_id for c in rec.recommend(ctx)]
      for name, rec in recommenders.items()
    }
    if not all(ranked.values()):
      continue
    pool = len(ranked[order[0]])
    pool_sizes.append(pool)
    rows[user_id] = {
      name: _metrics(ids, relevant, k_top, k_recall, pool)
      for name, ids in ranked.items()
    }
  return _tables(rows, order, pool_sizes)


def paired_bootstrap(
  per_user: pd.DataFrame,
  a: str,
  b: str,
  metric: str = "NDCG@10",
  n_boot: int = 2000,
  seed: int = 42,
) -> dict:
  """Mean per-user difference ``b - a`` in ``metric`` with a 95% bootstrap CI.

  Users are resampled with replacement and both orderings move together, so
  the interval reflects the paired comparison, not two independent means.
  Also counts the users for whom ``b`` is better, worse, or the same.
  """
  diff = (per_user[(b, metric)] - per_user[(a, metric)]).to_numpy()
  rng = np.random.default_rng(seed)
  boot = np.array([rng.choice(diff, diff.size).mean() for _ in range(n_boot)])
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return {
    "diff": float(diff.mean()),
    "ci_low": float(lo),
    "ci_high": float(hi),
    "better": int((diff > 0).sum()),
    "worse": int((diff < 0).sum()),
    "same": int((diff == 0).sum()),
  }


def validation_metric(
  candidates: dict[str, list[ScoredItem]],
  relevance: dict[str, set],
  make_scorer,
  metric: str = "NDCG@10",
  score_key: str = "cross_encoder",
):
  """``val_fn`` for ``fit_cross_encoder``: the rescored ordering's ``metric``.

  ``make_scorer(model)`` wraps the model being trained as a pipeline scorer,
  so validation runs through exactly the scorer the test evaluation uses.
  """

  def val_fn(model) -> float:
    summary, _ = evaluate_orderings(
      candidates, relevance, {RESCORED_ROW: (make_scorer(model), score_key)}
    )
    return float(summary.loc[RESCORED_ROW, metric])

  return val_fn
