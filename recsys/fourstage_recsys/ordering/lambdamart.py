"""LambdaMART ordering model (chapter 7, sections 7.4.2-7.4.3).

Trees trained with a listwise loss: LightGBM's lambdarank objective over
per-user groups. The group array is the one thing to get right -- rows
per user must be contiguous, one count per user, counts summing to the
number of rows. train_lambdamart enforces that by sorting and counting.

For the cross-feature ablation, call train_lambdamart twice: once with
features.FEATURE_COLS and once with FEATURE_COLS_NO_CROSS. For the
loss-vs-architecture comparison, train_pointwise_gbdt gives the same
trees with a pointwise (binary log-loss) objective.

Note on the top-k focus: `eval_at` only affects metric reporting, and
only when an eval_set is passed. What restricts lambdarank's pairs to
the top of the list is `lambdarank_truncation_level` (LightGBM default 30).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import lightgbm as lgb
import numpy as np
import pandas as pd

GBDT_PARAMS = dict(
  n_estimators=600,
  learning_rate=0.05,
  num_leaves=63,
  min_child_samples=100,
  importance_type="gain",
  verbose=-1,
)


@contextmanager
def _quiet_eval_set():
  """LightGBM >= 4.7 renames eval_set to eval_X/eval_y; we keep eval_set so
  every 4.x release works, and silence the rename warning."""
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
    yield


def train_lambdamart(
  train_frame: pd.DataFrame,
  feature_cols: list[str],
  n_estimators: int = 600,
  truncation_level: int = 30,
  seed: int = 7,
  eval_frame: pd.DataFrame | None = None,
  callbacks: list | None = None,
) -> lgb.LGBMRanker:
  """Listing 7.2: fit a LambdaMART ranker on per-user candidate groups.

  Pass `eval_frame` (held-out users) to track NDCG@10 per boosting round
  on both the training rows and those users; `callbacks` receive them
  (see iteration_logger). Monitoring only: nothing is early-stopped.
  """
  train_frame = train_frame.sort_values("userId", kind="stable")  # A
  group_sizes = train_frame.groupby("userId", sort=True).size().to_numpy()  # B

  ranker = lgb.LGBMRanker(
    objective="lambdarank",
    **{**GBDT_PARAMS, "n_estimators": n_estimators},
    label_gain=[0, 1],  # C
    lambdarank_truncation_level=truncation_level,  # D
    random_state=seed,
  )
  X, y = train_frame[feature_cols], train_frame["relevant"]
  eval_kw = {}
  if eval_frame is not None:
    eval_frame = eval_frame.sort_values("userId", kind="stable")
    eval_kw = dict(
      eval_set=[(X, y), (eval_frame[feature_cols], eval_frame["relevant"])],
      eval_group=[group_sizes, eval_frame.groupby("userId", sort=True).size().to_numpy()],
      eval_names=["train", "val"],
      eval_at=[10],
    )
  with _quiet_eval_set():
    ranker.fit(
      X,
      y,
      group=group_sizes,  # E
      callbacks=callbacks,
      **eval_kw,
    )
  return ranker


# A Rows for each user must be contiguous before we count them
# B One count per user; the entries sum to len(train_frame)
# C Binary relevance, so two gains are enough
# D Only pairs involving the top of the list get lambda gradients
# E Tells LightGBM where each user's candidate list starts and ends


class _RawScore:
  """predict() returns the raw margin, so a classifier ranks like a ranker."""

  def __init__(self, model):
    self.model = model

  def predict(self, X):
    return self.model.predict(X, raw_score=True)


def train_pointwise_gbdt(
  train_frame: pd.DataFrame,
  feature_cols: list[str],
  n_estimators: int = 600,
  seed: int = 7,
  eval_frame: pd.DataFrame | None = None,
  callbacks: list | None = None,
):
  """Same trees, pointwise loss: the missing corner of the loss x architecture grid.

  `eval_frame` adds per-round binary log loss on the training rows and on
  those held-out rows, as in train_lambdamart.
  """
  clf = lgb.LGBMClassifier(
    objective="binary",
    **{**GBDT_PARAMS, "n_estimators": n_estimators},
    random_state=seed,
  )
  X, y = train_frame[feature_cols], train_frame["relevant"]
  eval_kw = {}
  if eval_frame is not None:
    eval_kw = dict(
      eval_set=[(X, y), (eval_frame[feature_cols], eval_frame["relevant"])],
      eval_names=["train", "val"],
    )
  with _quiet_eval_set():
    clf.fit(X, y, callbacks=callbacks, **eval_kw)
  return _RawScore(clf)


def iteration_logger(on_iteration, every: int = 10):
  """LightGBM callback: on_iteration(step, metrics) every `every` rounds.

  Metric names come out as "<set>_<metric>", e.g. "val_ndcg_at_10" or
  "train_binary_logloss", the same naming the DCN-v2 curves use, so the
  notebook can send both to MLflow with one function. The last round is
  always reported.
  """
  def _callback(env):
    step = env.iteration + 1
    if step % every and step != env.end_iteration:
      return
    on_iteration(step, {f"{data}_{name.replace('@', '_at_')}": float(value)
                        for data, name, value, *_ in env.evaluation_result_list})
  _callback.order = 30
  return _callback


def explain_ranker(
  model,
  sample: pd.DataFrame,
  feature_cols: list[str],
  center_by_user: bool = True,
  plot: bool = True,
):
  """Listing 7.4: SHAP values and beeswarm for the tree ranker.

  Call on a held-out sample. With center_by_user=True, each feature's
  SHAP values are centered within the user's candidate list. A ranker's
  order only depends on differences between a user's candidates, so a
  user-level feature (constant across the list) can shift every score
  without changing the order; centering removes that offset and leaves
  the part of each attribution that actually reorders the list.
  """
  import shap

  explainer = shap.TreeExplainer(getattr(model, "model", model))
  values = explainer.shap_values(sample[feature_cols])
  if isinstance(values, list):  # older shap, binary classifier
    values = values[-1]
  if center_by_user:
    frame = pd.DataFrame(values, index=sample.index, columns=feature_cols)
    values = (frame - frame.groupby(sample["userId"]).transform("mean")).to_numpy()
  if plot:
    shap.summary_plot(values, sample[feature_cols])
  return values


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
  """Gain-based importances as a tidy frame, for quick sanity checks."""
  m = getattr(model, "model", model)
  return (
    pd.DataFrame({"feature": feature_cols, "gain": m.feature_importances_})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
  )


def lift_curve(
  fit_frame: pd.DataFrame,
  test_frame: pd.DataFrame,
  feature_cols: list[str],
  fractions=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
  n_relevant: dict | None = None,
  k: int = 10,
  seed: int = 7,
) -> pd.DataFrame:
  """NDCG lift over the scored order as a function of training users.

  The traffic argument made concrete: how much logged data does the
  ranker need before it adds anything on top of the upstream score?
  """
  from .evaluation import ScoredOrderBaseline, evaluate_ranker, paired_bootstrap

  _, base = evaluate_ranker(
    test_frame,
    feature_cols,
    ScoredOrderBaseline(),
    k=k,
    n_relevant=n_relevant,
    return_per_user=True,
  )
  users = np.array(sorted(fit_frame["userId"].unique()))
  rng = np.random.default_rng(seed)
  rng.shuffle(users)
  rows = []
  for frac in fractions:
    keep = set(users[: max(int(len(users) * frac), 2)].tolist())
    sub = fit_frame[fit_frame["userId"].isin(keep)]
    model = train_lambdamart(sub, feature_cols, seed=seed)
    _, pu = evaluate_ranker(
      test_frame, feature_cols, model, k=k, n_relevant=n_relevant, return_per_user=True
    )
    ci = paired_bootstrap(base[f"ndcg@{k}"], pu[f"ndcg@{k}"], seed=seed)
    rows.append(
      {
        "fraction": frac,
        "train_users": len(keep),
        f"ndcg@{k}": pu[f"ndcg@{k}"].mean(),
        "lift": ci["diff"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
      }
    )
  return pd.DataFrame(rows)
