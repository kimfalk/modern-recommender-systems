"""Selection-honest model comparison for chapter 7.

Every model family is tuned the same way, and none of it touches the test
frame. Selection runs forward in time, on the chapter's split shifted one
slice back (splits.TUNING_CUTS):

    tuning:  train on tune_labels (60-70%), validate on mid (70-80%)
    final:   train on mid (70-80%),         test on late (80-100%)

1. Each configuration is trained on the tuning rows (upstream fit on each
   user's first 60%, labels from 60-70%) and scored with NDCG@10 on the
   ranker's training rows (labels from 70-80%) -- the next window, exactly
   as the final model is scored on the window after its training labels.
2. The training length is selected in the same pass: trees are scored at
   n_estimators checkpoints, DCN-v2 after every epoch. Stochastic
   configurations are averaged over three seeds before the best
   (configuration, checkpoint) is picked.
3. The selected configurations are written to JSON before any model sees
   the test frame. The final models are refit on the ranker's training
   rows and evaluated on the test frame once.

The per-job results are cached as JSON, keyed by family, configuration
and seed, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import per_user_ndcg
from .features import (
  CROSS_COLS,
  DCN_DENSE_COLS,
  FEATURE_COLS,
  FEATURE_COLS_NO_CROSS,
  attach_labels,
  attach_upstream_scores,
  build_feature_frame,
  build_genre_matrix,
)
from .splits import feature_history, request_times

GRID_SEED = 7  # samples the tree configurations
SEEDS = (7, 11, 13)
K = 10

# Graded relevance for the graded-label LambdaMART. The held-out positives
# are ratings >= 4.0 only, so the grades split those; everything that is
# not a held-out positive is 0. Evaluation stays binary.
RATING_GRADES = {4.0: 1, 4.5: 2, 5.0: 3}
GRADED_LABEL_GAIN = (0, 1, 3, 7)

# Trees: learning rate fixed, n_estimators selected from these checkpoints.
TREE_LEARNING_RATE = 0.05
TREE_CHECKPOINTS = tuple(range(50, 1001, 50))
TREE_GRID = {
  "num_leaves": (31, 63, 127),
  "min_child_samples": (20, 100, 400),
  "reg_lambda": (0.0, 10.0),
  "colsample_bytree": (1.0, 0.7),
}
TRUNCATION_ALL = 10_000  # longer than any candidate list: every pair counts
TRUNCATION_LEVELS = (10, 30, TRUNCATION_ALL)
DEFAULT_TREE = {
  "num_leaves": 63,
  "min_child_samples": 100,
  "reg_lambda": 0.0,
  "colsample_bytree": 1.0,
}

DCN_EPOCHS = 10
DCN_BASE = {
  "n_cross": 3,
  "mlp_dims": [128, 64],
  "lr": 1e-3,
  "batch_size": 1024,
  "weight_decay": 0.1,
  "mlp_dropout": 0.0,
  "cross_rank": None,
  "use_crosses": False,
}

FAMILIES = {
  "lambdamart_x": {
    "kind": "tree",
    "objective": "lambdarank",
    "features": FEATURE_COLS,
    "label": "relevant",
  },
  "lambdamart_nox": {
    "kind": "tree",
    "objective": "lambdarank",
    "features": FEATURE_COLS_NO_CROSS,
    "label": "relevant",
  },
  "gbdt_pointwise": {
    "kind": "tree",
    "objective": "binary",
    "features": FEATURE_COLS,
    "label": "relevant",
  },
  "lambdamart_graded": {
    "kind": "tree",
    "objective": "lambdarank",
    "features": FEATURE_COLS,
    "label": "grade",
  },
  "dcn": {"kind": "dcn"},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def attach_graded_labels(frame: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
  """Add `grade` (0-3) from the held-out ratings; non-positives get 0."""
  graded = (
    heldout.groupby(["userId", "movieId"], as_index=False)["rating"]
    .max()
    .assign(grade=lambda d: d["rating"].map(RATING_GRADES))
  )
  if graded["grade"].isna().any():
    raise ValueError(f"ratings outside {sorted(RATING_GRADES)} in the held-out slice")
  out = frame.merge(
    graded[["userId", "movieId", "grade"]], on=["userId", "movieId"], how="left"
  )
  out["grade"] = out["grade"].fillna(0).astype(int)
  if not (out["grade"].gt(0) == out["relevant"].gt(0)).all():
    raise ValueError("graded labels disagree with the binary relevance labels")
  return out


def load_ordering_data(art_dir: Path) -> dict:
  """The chapter-7 training and test frames, built as in ch07_ordering.ipynb.

  Training rows: features from history before the mid and late slices,
  labels from mid. Test rows: history before late, labels from late.
  Tuning rows (scripts/ch07_export_tuning_candidates.py): upstream fit on
  each user's first 60%, history before the 60-70% slice, labels from it.
  """
  art_dir = Path(art_dir)
  ratings = pd.read_parquet(art_dir / "ratings_sample.parquet")
  movies = pd.read_parquet(art_dir / "movies.parquet")
  positives = pd.read_parquet(art_dir / "positives.parquet")
  candidates_fit = pd.read_parquet(art_dir / "candidates_fit.parquet")
  candidates_test = pd.read_parquet(art_dir / "candidates_test.parquet")
  candidates_tune = pd.read_parquet(art_dir / "candidates_tune.parquet")
  tune_labels = pd.read_parquet(art_dir / "tuning_labels.parquet")
  mid = positives[positives["slice"] == "mid"]
  late = positives[positives["slice"] == "late"]
  genre_matrix = build_genre_matrix(movies)

  def make_frame(candidates, labels, future):
    history = feature_history(ratings, future)
    rows = attach_upstream_scores(
      attach_labels(candidates[["userId", "movieId"]], labels), candidates
    )
    return build_feature_frame(
      rows, history, genre_matrix, request_ts=request_times(future)
    )

  fit_frame = make_frame(candidates_fit, mid, pd.concat([mid, late]))
  fit_frame = attach_graded_labels(fit_frame, mid)
  test_frame = make_frame(candidates_test, late, late)
  tune_frame = make_frame(
    candidates_tune, tune_labels, pd.concat([tune_labels, mid, late])
  )
  tune_frame = attach_graded_labels(tune_frame, tune_labels)
  for frame in (tune_frame, fit_frame, test_frame):
    if frame[FEATURE_COLS].isna().any().any():
      raise ValueError("missing feature values in the ordering frames")
  return {
    "tune_frame": tune_frame.reset_index(drop=True),
    "fit_frame": fit_frame.reset_index(drop=True),
    "test_frame": test_frame.reset_index(drop=True),
    "n_relevant_fit": mid.groupby("userId").size().to_dict(),
    "n_relevant_test": late.groupby("userId").size().to_dict(),
  }


# ---------------------------------------------------------------------------
# Search spaces (~12 configurations per family)
# ---------------------------------------------------------------------------


def tree_configs(objective: str, n: int = 12, seed: int = GRID_SEED) -> list[dict]:
  """The current default plus n-1 distinct configurations sampled from TREE_GRID."""
  default = dict(DEFAULT_TREE)
  axes = dict(TREE_GRID)
  if objective == "lambdarank":
    default["truncation_level"] = 30
    axes["truncation_level"] = TRUNCATION_LEVELS
  keys = list(axes)
  grid = [dict(zip(keys, values)) for values in itertools.product(*axes.values())]
  grid = [c for c in grid if c != default]
  picks = np.random.default_rng(seed).choice(len(grid), size=n - 1, replace=False)
  return [default] + [grid[i] for i in sorted(picks)]


def dcn_configs() -> list[dict]:
  """Embedding regularization factorial (8) + MLP dropout, low rank, crosses (4)."""
  configs = []
  for emb_dim, emb_dropout, emb_wd in itertools.product(
    (8, 16), (0.0, 0.2), (0.1, 1.0)
  ):
    configs.append(
      {"emb_dim": emb_dim, "emb_dropout": emb_dropout, "emb_weight_decay": emb_wd}
    )
  regularized = {"emb_dim": 8, "emb_dropout": 0.2, "emb_weight_decay": 1.0}
  configs += [
    {**regularized, "mlp_dropout": 0.2},
    {**regularized, "cross_rank": 8},
    {**regularized, "use_crosses": True},
    {"emb_dim": 16, "emb_dropout": 0.0, "emb_weight_decay": 0.1, "use_crosses": True},
  ]
  return [{**DCN_BASE, **c} for c in configs]


def is_stochastic(family: str, config: dict) -> bool:
  if FAMILIES[family]["kind"] == "dcn":
    return True
  return config.get("colsample_bytree", 1.0) < 1.0


def config_id(config: dict) -> str:
  blob = json.dumps(config, sort_keys=True, default=str)
  return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_tree(
  frame,
  family: str,
  config: dict,
  n_estimators: int,
  seed: int,
  n_jobs: int = -1,
  feature_cols=None,
):
  """Fit one tree model of a family; returns the LightGBM booster."""
  import lightgbm as lgb

  spec = FAMILIES[family]
  cols = list(spec["features"] if feature_cols is None else feature_cols)
  params = {k: v for k, v in config.items() if k != "truncation_level"}
  common = dict(
    n_estimators=n_estimators,
    learning_rate=TREE_LEARNING_RATE,
    random_state=seed,
    n_jobs=n_jobs,
    verbose=-1,
    **params,
  )
  if spec["objective"] == "lambdarank":
    frame = frame.sort_values("userId", kind="stable")
    groups = frame.groupby("userId", sort=True).size().to_numpy()
    gain = list(GRADED_LABEL_GAIN) if spec["label"] == "grade" else [0, 1]
    model = lgb.LGBMRanker(
      objective="lambdarank",
      label_gain=gain,
      lambdarank_truncation_level=config["truncation_level"],
      **common,
    )
    model.fit(frame[cols], frame[spec["label"]], group=groups)
  else:
    model = lgb.LGBMClassifier(objective="binary", **common)
    model.fit(frame[cols], frame[spec["label"]])
  return model.booster_


def tree_scores(booster, frame, feature_cols, n_estimators: int) -> np.ndarray:
  return booster.predict(
    frame[list(feature_cols)], num_iteration=n_estimators, raw_score=True
  )


def tree_score_checkpoints(booster, frame, feature_cols, checkpoints):
  """Yield (n_estimators, raw scores) at each checkpoint, trees added in chunks.

  Raw scores are a sum over trees, so each checkpoint only predicts the
  trees added since the previous one.
  """
  X = frame[list(feature_cols)]
  total, done = np.zeros(len(frame)), 0
  for n in checkpoints:
    total += booster.predict(
      X, start_iteration=done, num_iteration=n - done, raw_score=True
    )
    done = n
    yield n, total.copy()


def dcn_columns(config: dict) -> list[str]:
  return list(FEATURE_COLS if config["use_crosses"] else DCN_DENSE_COLS)


def fit_dcn(
  train_frame,
  config: dict,
  n_movies: int,
  epochs: int,
  seed: int,
  eval_frame=None,
  n_relevant=None,
):
  """Train DCN-v2 for a fixed number of epochs; optionally score every epoch.

  With `eval_frame`, returns the per-epoch history including NDCG@10 on
  that frame (the validation curve). Returns (DCNRanker, history).
  """
  import torch
  from torch.utils.data import DataLoader

  from .dataset import CandidateDataset, fit_scaler
  from .dcn import DCNRanker, DCNv2, train_dcn

  cols = dcn_columns(config)
  scaler = fit_scaler(train_frame, cols)
  loader = DataLoader(
    CandidateDataset(train_frame, cols, scaler),
    batch_size=config["batch_size"],
    shuffle=True,
  )
  torch.manual_seed(seed)  # weights and batch order
  model = DCNv2(
    n_movies,
    len(cols),
    emb_dim=config["emb_dim"],
    n_cross=config["n_cross"],
    mlp_dims=tuple(config["mlp_dims"]),
    emb_dropout=config["emb_dropout"],
    mlp_dropout=config["mlp_dropout"],
    cross_rank=config["cross_rank"],
  )

  eval_fn = None
  if eval_frame is not None:

    def eval_fn(m):
      scores = DCNRanker(m, scaler, cols).predict(eval_frame[cols + ["movie_idx"]])
      ndcg = per_user_ndcg(eval_frame, scores, K, n_relevant)
      return {"ndcg": float(ndcg.mean())}

  model, history = train_dcn(
    model,
    loader,
    epochs=epochs,
    lr=config["lr"],
    weight_decay=config["weight_decay"],
    emb_weight_decay=config["emb_weight_decay"],
    eval_fn=eval_fn,
    select_on=None,
  )
  return DCNRanker(model, scaler, cols), history


# ---------------------------------------------------------------------------
# Jobs (run in worker processes by scripts/ch07_selection.py)
# ---------------------------------------------------------------------------

_DATA: dict = {}


def init_worker(frames_path: str, n_threads: int) -> None:
  """Process-pool initializer: load the cached frames once per worker."""
  import torch

  torch.set_num_threads(n_threads)
  _DATA.update(pd.read_pickle(frames_path))
  _DATA["n_threads"] = n_threads


def tune_job(family: str, config: dict, seed: int) -> dict:
  """Train on the tuning rows, score the next window at every checkpoint."""
  train, valid = _DATA["tune_frame"], _DATA["fit_frame"]
  n_rel = _DATA["n_relevant_fit"]
  if FAMILIES[family]["kind"] == "tree":
    booster = fit_tree(
      train, family, config, max(TREE_CHECKPOINTS), seed, n_jobs=_DATA["n_threads"]
    )
    curve = [
      float(per_user_ndcg(valid, scores, K, n_rel).mean())
      for _, scores in tree_score_checkpoints(
        booster, valid, FAMILIES[family]["features"], TREE_CHECKPOINTS
      )
    ]
    return {"checkpoints": list(TREE_CHECKPOINTS), "ndcg": curve}
  _, history = fit_dcn(
    train,
    config,
    _DATA["n_movies"],
    DCN_EPOCHS,
    seed,
    eval_frame=valid,
    n_relevant=n_rel,
  )
  return {
    "checkpoints": list(range(1, DCN_EPOCHS + 1)),
    "ndcg": [h["ndcg"] for h in history],
    "train_loss": [h["train_loss"] for h in history],
  }


def final_job(
  family: str, config: dict, checkpoint: int, seed: int, feature_cols=None
) -> np.ndarray:
  """Refit on the ranker's training rows with the selected configuration; score test.

  `feature_cols` overrides a tree family's features (the per-cross ablation).
  """
  train, test = _DATA["fit_frame"], _DATA["test_frame"]
  if FAMILIES[family]["kind"] == "tree":
    cols = FAMILIES[family]["features"] if feature_cols is None else feature_cols
    booster = fit_tree(
      train,
      family,
      config,
      checkpoint,
      seed,
      n_jobs=_DATA["n_threads"],
      feature_cols=cols,
    )
    return tree_scores(booster, test, cols, checkpoint)
  ranker, _ = fit_dcn(train, config, _DATA["n_movies"], checkpoint, seed)
  return ranker.predict(test[dcn_columns(config) + ["movie_idx"]])


class JsonCache:
  """One JSON file per finished job, so reruns skip work already done."""

  def __init__(self, root: Path):
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)

  def path(self, *parts) -> Path:
    return self.root / ("_".join(str(p) for p in parts) + ".json")

  def get(self, *parts):
    p = self.path(*parts)
    return json.loads(p.read_text()) if p.exists() else None

  def put(self, value, *parts) -> None:
    p = self.path(*parts)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(p)


def tune_jobs(family: str) -> list[tuple]:
  """(family, config_id, config, seed) for every job a family needs."""
  configs = (
    dcn_configs() if family == "dcn" else tree_configs(FAMILIES[family]["objective"])
  )
  return [
    (family, config_id(config), config, seed)
    for config in configs
    for seed in (SEEDS if is_stochastic(family, config) else SEEDS[:1])
  ]


def summarize_tuning(family: str, results: dict) -> pd.DataFrame:
  """Validation NDCG@10 per (config, checkpoint), mean over seeds.

  `results[(config_id, seed)]` holds one job's curve.
  """
  rows = [
    {"family": family, "config_id": cid, "seed": seed, "checkpoint": chk, "ndcg": ndcg}
    for (cid, seed), res in results.items()
    for chk, ndcg in zip(res["checkpoints"], res["ndcg"])
  ]
  summary = (
    pd.DataFrame(rows)
    .groupby(["config_id", "checkpoint"])["ndcg"]
    .agg(["mean", "std", "count"])
    .reset_index()
  )
  return summary.rename(
    columns={"mean": "val_ndcg", "std": "seed_std", "count": "n_seeds"}
  )


def select_config(family: str, results: dict) -> dict:
  """Best (configuration, checkpoint) by validation NDCG@10 (mean over seeds)."""
  summary = summarize_tuning(family, results)
  best = summary.sort_values(
    ["val_ndcg", "config_id", "checkpoint"], ascending=[False, True, True]
  ).iloc[0]
  return {
    "config_id": best["config_id"],
    "checkpoint": int(best["checkpoint"]),
    "val_ndcg": float(best["val_ndcg"]),
    "seed_std": None if pd.isna(best["seed_std"]) else float(best["seed_std"]),
    "n_seeds": int(best["n_seeds"]),
  }


__all__ = [
  "CROSS_COLS",
  "DEFAULT_TREE",
  "FAMILIES",
  "GRID_SEED",
  "SEEDS",
  "TREE_CHECKPOINTS",
  "JsonCache",
  "attach_graded_labels",
  "config_id",
  "dcn_columns",
  "dcn_configs",
  "final_job",
  "fit_dcn",
  "fit_tree",
  "init_worker",
  "is_stochastic",
  "load_ordering_data",
  "select_config",
  "summarize_tuning",
  "tree_configs",
  "tree_score_checkpoints",
  "tree_scores",
  "tune_job",
  "tune_jobs",
]
