"""Final-model evaluation and the predeclared comparisons (chapter 7).

Each final model is evaluated on the test frame once per seed. The table
row is the mean over seeds; per-user NDCG is averaged over seeds the same
way, so the paired bootstrap compares expected performance rather than one
lucky seed. Only PREDECLARED comparisons get a confidence interval in the
chapter; everything else is described in prose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import (
  PrecomputedScores,
  ScoredOrderBaseline,
  evaluate_ranker,
  paired_bootstrap,
)
from .ranker_distributions import first_relevant_rank

SCORED_ORDER = "scored_order"
LEARNED = (
  "lambdamart_x",
  "lambdamart_nox",
  "gbdt_pointwise",
  "dcn",
  "lambdamart_graded",
)

# (a, b): the reported difference is b - a.
PREDECLARED = (
  *((SCORED_ORDER, m) for m in LEARNED),  # 1. each learned ranker vs scored order
  ("lambdamart_nox", "lambdamart_x"),  # 2. hand-built crosses
  ("lambdamart_nox", "dcn"),  # 3. DCN-v2 vs its fair tree comparison
  ("lambdamart_x", "dcn"),  # 4. DCN-v2 vs LambdaMART with crosses
  ("lambdamart_x", "gbdt_pointwise"),  # 5. loss: pointwise vs listwise, same trees
  ("dcn", "gbdt_pointwise"),  # 6. model: trees vs network
)


def evaluate_seeds(
  test_frame: pd.DataFrame, scores_by_seed: dict, n_relevant: dict, k: int = 10
) -> dict:
  """Metrics for one final model given its test scores per seed."""
  per_seed, per_user, ranks = {}, [], {}
  for seed, scores in scores_by_seed.items():
    s = pd.Series(np.asarray(scores, dtype=float), index=test_frame.index)
    res, pu = evaluate_ranker(
      test_frame,
      ["cross_encoder_score"],
      PrecomputedScores(s),
      k=k,
      n_relevant=n_relevant,
      return_per_user=True,
    )
    per_seed[seed] = res
    per_user.append(pu[f"ndcg@{k}"])
    ranks[seed] = first_relevant_rank(test_frame, s.to_numpy())
  metrics = pd.DataFrame(per_seed).T
  ndcg = metrics[f"ndcg@{k}"]
  return {
    "metrics": metrics.mean().to_dict(),
    "ndcg_by_seed": {int(s): float(v) for s, v in ndcg.items()},
    "seed_range": [float(ndcg.min()), float(ndcg.max())],
    "per_user": sum(per_user) / len(per_user),
    "first_relevant_rank": ranks,
  }


def scored_order_scores(test_frame: pd.DataFrame) -> dict:
  return {0: ScoredOrderBaseline().predict(test_frame)}


def compare(
  per_user: dict, pairs=PREDECLARED, n_boot: int = 2000, seed: int = 7
) -> list:
  """Paired bootstrap for each (a, b) pair; difference is b - a."""
  out = []
  for a, b in pairs:
    if a in per_user and b in per_user:
      ci = paired_bootstrap(per_user[a], per_user[b], n_boot=n_boot, seed=seed)
      out.append({"a": a, "b": b, **ci})
  return out
