"""Chapter 7: selection-honest ranker comparison (forward-in-time tuning).

Phases, each resumable from the cache:
  tune    train every configuration of every family on the tuning rows (labels
          60-70%) and score the next window (labels 70-80%)
  select  pick (configuration, n_estimators/epochs) per family -> JSON, before any
          test evaluation
  final   refit the selected configurations on the ranker's training rows (labels
          70-80%), score the test frame (80-100%) once, plus the per-cross
          ablation, then compare

Needs candidates_tune.parquet and tuning_labels.parquet
(scripts/ch07_export_tuning_candidates.py). Run from the repository root:
    python scripts/ch07_selection.py --phase all
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.fourstage_recsys.ordering import comparison as cmp
from recsys.fourstage_recsys.ordering import selection as sel
from recsys.fourstage_recsys.ordering.dataset import (
  add_movie_index,
  make_movie_index,
)


def repo_relative(path: Path) -> str:
  """Path as written into result files: relative to the repo root, so no
  machine-specific prefix gets committed. Paths outside the repo stay absolute."""
  path = path.resolve()
  return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


TREE_FAMILIES = [
  "lambdamart_x",
  "lambdamart_nox",
  "gbdt_pointwise",
  "lambdamart_graded",
]

# What ch07_ordering.ipynb runs today, to flag where tuning picks something else.
NOTEBOOK_DEFAULTS = {
  "tree": {**sel.DEFAULT_TREE, "n_estimators": 600},
  "lambdarank": {"truncation_level": 30},
  "dcn": {**sel.DCN_BASE, "emb_dim": 8, "emb_dropout": 0.2, "emb_weight_decay": 1.0},
}


def log(msg: str) -> None:
  print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def prepare_frames(art_dir: Path, cache_dir: Path) -> Path:
  path = cache_dir / "frames.pkl"
  if path.exists():
    return path
  log("building tuning, training and test frames")
  data = sel.load_ordering_data(art_dir)
  names = ("tune_frame", "fit_frame", "test_frame")
  movie_index = make_movie_index(pd.concat([data[n]["movieId"] for n in names]))
  for n in names:
    data[n] = add_movie_index(data[n], movie_index)
  data["n_movies"] = len(movie_index)
  pd.to_pickle(data, path)
  log(
    f"frames cached: {len(data['tune_frame']):,} tuning, "
    f"{len(data['fit_frame']):,} training, {len(data['test_frame']):,} test rows"
  )
  return path


def run_pool(fn, jobs: list, frames: Path, workers: int, threads: int, on_done) -> None:
  if not jobs:
    return
  ctx = mp.get_context("spawn")
  with ProcessPoolExecutor(
    workers,
    mp_context=ctx,
    initializer=sel.init_worker,
    initargs=(str(frames), threads),
  ) as pool:
    futures = {pool.submit(fn, *args): key for key, args in jobs}
    t0 = time.time()
    for i, fut in enumerate(as_completed(futures), 1):
      on_done(futures[fut], fut.result())
      eta = (time.time() - t0) / i * (len(jobs) - i)
      log(f"  {i}/{len(jobs)} done ({futures[fut]}), ~{eta / 60:.0f} min left")


def phase_tune(families, frames, cache, args) -> None:
  for family in families:
    jobs = sel.tune_jobs(family)
    pending = [
      ((family, cid, seed), (family, config, seed))
      for _, cid, config, seed in jobs
      if cache.get("tune", family, cid, seed) is None
    ]
    log(f"tuning {family}: {len(jobs)} jobs, {len(pending)} to run")
    tree = sel.FAMILIES[family]["kind"] == "tree"
    run_pool(
      sel.tune_job,
      pending,
      frames,
      args.tree_workers if tree else args.dcn_workers,
      args.tree_threads if tree else args.dcn_threads,
      lambda key, res: cache.put(res, "tune", *key),
    )


def tuning_results(family, cache) -> tuple[dict, dict]:
  results, configs = {}, {}
  for _, cid, config, seed in sel.tune_jobs(family):
    res = cache.get("tune", family, cid, seed)
    if res is None:
      raise RuntimeError(f"missing tuning result {family} {cid} seed {seed}")
    results[(cid, seed)] = res
    configs[cid] = config
  return results, configs


def differs_from_notebook(family: str, config: dict, checkpoint: int) -> dict:
  if family == "dcn":
    ref = NOTEBOOK_DEFAULTS["dcn"]
    diff = {
      k: {"selected": config[k], "notebook": ref[k]}
      for k in ref
      if config.get(k) != ref[k]
    }
    return diff  # the notebook picks DCN epochs on its own validation split
  ref = dict(NOTEBOOK_DEFAULTS["tree"])
  if sel.FAMILIES[family]["objective"] == "lambdarank":
    ref.update(NOTEBOOK_DEFAULTS["lambdarank"])
  chosen = {**config, "n_estimators": checkpoint}
  return {
    k: {"selected": chosen[k], "notebook": ref[k]}
    for k in ref
    if chosen.get(k) != ref[k]
  }


def phase_select(families, cache, out_path: Path) -> None:
  selected, summaries = {}, []
  for family in families:
    results, configs = tuning_results(family, cache)
    best = sel.select_config(family, results)
    config = configs[best["config_id"]]
    summary = sel.summarize_tuning(family, results)
    summaries.append(summary.assign(family=family))
    spec = sel.FAMILIES[family]
    selected[family] = {
      **best,
      "config": config,
      "training_length": ("n_estimators" if spec["kind"] == "tree" else "epochs"),
      "features": (
        sel.dcn_columns(config) if spec["kind"] == "dcn" else list(spec["features"])
      ),
      "label": spec.get("label", "relevant"),
      "seeds": list(sel.SEEDS if sel.is_stochastic(family, config) else sel.SEEDS[:1]),
      "n_configs": len(configs),
      "differs_from_notebook": differs_from_notebook(
        family, config, best["checkpoint"]
      ),
    }
    log(
      f"selected {family}: {best['config_id']} at {best['checkpoint']} "
      f"(validation NDCG@10 {best['val_ndcg']:.4f}, {best['n_seeds']} seed(s))"
    )
  payload = {
    "written_at": datetime.now(UTC).isoformat(),
    "procedure": {
      "tuning_rows": "upstream fit on 0-60%, labels 60-70% (candidates_tune.parquet)",
      "validation_rows": "the ranker's training rows: upstream fit on 0-70%, labels 70-80%",
      "selection_metric": "validation NDCG@10, mean over seeds for stochastic configs",
      "grid_seed": sel.GRID_SEED,
      "tree_learning_rate": sel.TREE_LEARNING_RATE,
      "tree_checkpoints": list(sel.TREE_CHECKPOINTS),
      "dcn_epochs": sel.DCN_EPOCHS,
    },
    "selected": selected,
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload, indent=2, default=str))
  pd.concat(summaries).to_csv(
    out_path.with_name("ch07_tuning_summary.csv"), index=False
  )
  log(f"wrote {out_path}")


def phase_final(selected_path: Path, frames, cache, args, results_dir: Path) -> None:
  selected = json.loads(selected_path.read_text())["selected"]
  final_dir = cache.root / "final"
  final_dir.mkdir(exist_ok=True)

  jobs = []  # (model name, seed) -> final_job args
  for family, s in selected.items():
    for seed in s["seeds"]:
      jobs.append(((family, seed), (family, s["config"], s["checkpoint"], seed)))
  lm = selected["lambdamart_x"]
  for cross in sel.CROSS_COLS:
    cols = [c for c in lm["features"] if c != cross]
    for seed in lm["seeds"]:
      jobs.append(
        (
          (f"lambdamart_x_minus_{cross}", seed),
          ("lambdamart_x", lm["config"], lm["checkpoint"], seed, cols),
        )
      )
  pending = [
    j for j in jobs if not (final_dir / f"{j[0][0]}_seed{j[0][1]}.npy").exists()
  ]
  log(f"final fits: {len(jobs)} jobs, {len(pending)} to run")
  run_pool(
    sel.final_job,
    pending,
    frames,
    args.final_workers,
    args.final_threads,
    lambda key, scores: np.save(final_dir / f"{key[0]}_seed{key[1]}.npy", scores),
  )

  data = pd.read_pickle(frames)
  test, n_rel = data["test_frame"], data["n_relevant_test"]
  scores = {}
  for (name, seed), _ in jobs:
    scores.setdefault(name, {})[seed] = np.load(final_dir / f"{name}_seed{seed}.npy")
  scores[cmp.SCORED_ORDER] = cmp.scored_order_scores(test)

  evals = {}
  for name, by_seed in scores.items():
    log(f"evaluating {name} ({len(by_seed)} seed(s))")
    evals[name] = cmp.evaluate_seeds(test, by_seed, n_rel)
  per_user = {name: e["per_user"] for name, e in evals.items()}

  base = per_user[cmp.SCORED_ORDER]
  table = {}
  for name, e in evals.items():
    vs_base = cmp.compare(per_user, [(cmp.SCORED_ORDER, name)])
    table[name] = {
      **e["metrics"],
      "ndcg_by_seed": e["ndcg_by_seed"],
      "seed_range": e["seed_range"],
      "vs_scored_order": vs_base[0] if vs_base and name != cmp.SCORED_ORDER else None,
    }
  ablation = cmp.compare(
    per_user, [("lambdamart_x", f"lambdamart_x_minus_{c}") for c in sel.CROSS_COLS]
  )
  graded = cmp.compare(
    per_user,
    [("gbdt_pointwise", "lambdamart_graded"), ("lambdamart_x", "lambdamart_graded")],
  )
  out = {
    "selected_configs_file": repo_relative(selected_path),
    "evaluated_at": datetime.now(UTC).isoformat(),
    "n_test_users": len(base),
    "table": table,
    "predeclared": cmp.compare(per_user),
    "exploratory": {"per_cross_ablation": ablation, "graded_labels": graded},
  }
  (results_dir / "ch07_final_results.json").write_text(
    json.dumps(out, indent=2, default=float)
  )
  ranks = pd.concat(
    [
      r.rename("first_relevant_rank").reset_index().assign(model=name, seed=seed)
      for name, e in evals.items()
      for seed, r in e["first_relevant_rank"].items()
    ]
  )
  ranks.to_parquet(results_dir / "ch07_first_relevant_rank.parquet", index=False)
  log(f"wrote {results_dir / 'ch07_final_results.json'} and first_relevant_rank data")


def main() -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--phase", choices=["tune", "select", "final", "all"], default="all")
  p.add_argument("--families", nargs="+", default=TREE_FAMILIES + ["dcn"])
  p.add_argument("--art-dir", type=Path, default=ROOT / "data/processed/chapter07")
  p.add_argument(
    "--cache-dir", type=Path, default=ROOT / "data/processed/chapter07/selection"
  )
  p.add_argument("--results-dir", type=Path, default=ROOT / "results")
  p.add_argument("--tree-workers", type=int, default=2)
  p.add_argument("--tree-threads", type=int, default=5)
  p.add_argument("--dcn-workers", type=int, default=5)
  p.add_argument("--dcn-threads", type=int, default=2)
  p.add_argument("--final-workers", type=int, default=2)
  p.add_argument("--final-threads", type=int, default=5)
  args = p.parse_args()

  cache = sel.JsonCache(args.cache_dir)
  frames = prepare_frames(args.art_dir, args.cache_dir)
  selected_path = args.results_dir / "ch07_selected_configs.json"
  if args.phase in ("tune", "all"):
    phase_tune(args.families, frames, cache, args)
  if args.phase in ("select", "all"):
    phase_select(args.families, cache, selected_path)
  if args.phase in ("final", "all"):
    phase_final(selected_path, frames, cache, args, args.results_dir)


if __name__ == "__main__":
  main()
