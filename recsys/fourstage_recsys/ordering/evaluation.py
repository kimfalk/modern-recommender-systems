"""Evaluation for the ordering stage (chapter 7).

The baseline is the scored order: candidates sorted by their chapter-5
cross-encoder score. Every model is measured against that row, and
list_divergence() reports how much the model actually changed the list
(overlap@k, changed@k, mean_shift@k) so a metric lift can't hide behind a
list that barely moved.

NDCG normalization: pass `n_relevant` (userId -> number of held-out
positives) to normalize by the user's full relevance set, exactly as the
chapter-5 metric does. The scored-order row then reproduces chapter 5's
cross-encoder NDCG@10 on the same users -- the wiring check. Without it,
the ideal DCG is computed from the relevant items that made it into the
candidate list (a candidate-conditional NDCG, always higher).

intra_list_diversity (Listing 7.8) lives here because it's a metric; the
list transforms it measures live in reranking.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ndcg_at_k(relevances: np.ndarray, k: int, n_relevant: int | None = None) -> float:
    """NDCG@k for a relevance array already in ranked order."""
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.sum() <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((rel * discounts[: rel.size]).sum())
    if n_relevant is None:
        ideal = np.sort(np.asarray(relevances, dtype=float))[::-1][:k]
    else:
        ideal = np.ones(min(int(n_relevant), k))
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevances: np.ndarray) -> float:
    """Reciprocal rank of the first relevant item; 0.0 if none."""
    rel = np.asarray(relevances, dtype=float)
    hits = np.flatnonzero(rel > 0)
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


def list_divergence(
    baseline_ids: np.ndarray, new_ids: np.ndarray, k: int = 10
) -> dict:
    """How much a reordering changed the list, against a baseline order.

    Both arrays must be permutations of the same candidate set.
    Returns:
      overlap@k    -- fraction of the top-k the two orderings share
      changed@k    -- number of top-k slots holding a different item
      mean_shift@k -- mean absolute rank shift of the new top-k items
    """
    baseline_ids = np.asarray(baseline_ids)
    new_ids = np.asarray(new_ids)
    top_b, top_n = baseline_ids[:k], new_ids[:k]
    kk = min(k, len(top_b), len(top_n))

    overlap = len(set(top_n.tolist()) & set(top_b.tolist())) / max(kk, 1)
    changed = int(sum(a != b for a, b in zip(top_n[:kk], top_b[:kk])))
    base_pos = {item: i for i, item in enumerate(baseline_ids.tolist())}
    shifts = [abs(i - base_pos[item]) for i, item in enumerate(top_n.tolist())
              if item in base_pos]
    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    return {
        f"overlap@{k}": overlap,
        f"changed@{k}": changed,
        f"mean_shift@{k}": mean_shift,
    }


def intra_list_diversity(item_ids, sim_matrix) -> float:
    """Listing 7.8: average pairwise dissimilarity of a list.

    `item_ids` index into `sim_matrix` -- for a per-candidate-list
    similarity matrix, pass positions (see ild_of_list).
    """
    k = len(item_ids)
    if k < 2:
        return 0.0
    total = sum(
        1.0 - sim_matrix[item_ids[a], item_ids[b]]
        for a in range(k) for b in range(a + 1, k)
    )
    return total / (k * (k - 1) / 2)


def ild_of_list(list_ids, cand_ids, sim_matrix) -> float:
    """ILD for a list of raw ids, given the candidate-local sim matrix."""
    pos = {item: i for i, item in enumerate(cand_ids)}
    return intra_list_diversity([pos[i] for i in list_ids], sim_matrix)


# ---------------------------------------------------------------------------
# Baselines as "models": anything with predict(X) -> scores
# ---------------------------------------------------------------------------

class ColumnScore:
    """Sort by one column. ColumnScore("cross_encoder_score") is the scored order."""

    def __init__(self, col: str):
        self.col = col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.col].to_numpy()


class ScoredOrderBaseline(ColumnScore):
    """The baseline row of every result table: the chapter-5 scored order."""

    def __init__(self):
        super().__init__("cross_encoder_score")


class LinearBlend:
    """score = sum_i w_i * standardized(col_i) -- the hand-tuned blend a ranker replaces."""

    def __init__(self, weights: dict, stats: dict):
        self.weights, self.stats = weights, stats

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(X))
        for col, w in self.weights.items():
            mu, sd = self.stats[col]
            out += w * (X[col].to_numpy() - mu) / sd
        return out


def fit_blend(
    fit_frame: pd.DataFrame,
    extra_col: str = "item_log_pop",
    score_col: str = "cross_encoder_score",
    grid=np.linspace(0.0, 2.0, 21),
    k: int = 10,
) -> tuple[LinearBlend, float]:
    """Tune one weight w in  score + w * extra  on the ranker's training rows."""
    stats = {c: (fit_frame[c].mean(), fit_frame[c].std() or 1.0)
             for c in (score_col, extra_col)}
    best_w, best = 0.0, -1.0
    for w in grid:
        model = LinearBlend({score_col: 1.0, extra_col: float(w)}, stats)
        val = evaluate_ranker(fit_frame, [score_col, extra_col], model, k=k)[f"ndcg@{k}"]
        if val > best:
            best_w, best = float(w), val
    return LinearBlend({score_col: 1.0, extra_col: best_w}, stats), best_w


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ranker(
    test_frame, feature_cols, model, k: int = 10,
    n_relevant: dict | None = None, return_per_user: bool = False,
):
    """Listing 7.3: score a ranker against the scored-order baseline.

    Keys match the chapter tables (ndcg@10, mrr, overlap@10, changed@10,
    mean_shift@10). With return_per_user=True, also returns a frame of
    per-user ndcg/mrr for paired bootstrap comparisons.
    """
    users, ndcgs, mrrs, divs = [], [], [], []
    for user_id, cand in test_frame.groupby("userId", sort=True):
        scored_order = cand.sort_values("cross_encoder_score", ascending=False,
                                        kind="stable")
        preds = model.predict(cand[feature_cols])
        ranked_order = cand.assign(_s=preds).sort_values("_s", ascending=False,
                                                         kind="stable")
        rel = ranked_order["relevant"].to_numpy()
        n_rel = None if n_relevant is None else n_relevant.get(user_id, 0)
        users.append(user_id)
        ndcgs.append(ndcg_at_k(rel, k, n_rel))
        mrrs.append(mrr(rel))
        divs.append(list_divergence(
            scored_order["movieId"].to_numpy(),
            ranked_order["movieId"].to_numpy(),
            k=k,
        ))
    out = {f"ndcg@{k}": float(np.mean(ndcgs)), "mrr": float(np.mean(mrrs))}
    for key in divs[0]:
        out[key] = float(np.mean([d[key] for d in divs]))
    if return_per_user:
        per_user = pd.DataFrame({"userId": users, f"ndcg@{k}": ndcgs, "mrr": mrrs})
        return out, per_user.set_index("userId")
    return out


def paired_bootstrap(
    per_user_a: pd.Series, per_user_b: pd.Series,
    n_boot: int = 2000, seed: int = 7, alpha: float = 0.05,
) -> dict:
    """Mean difference (b - a) over users with a bootstrap CI.

    Paired: the same users are resampled for both systems, which is what
    makes a difference of a few thousandths detectable at all.
    """
    a, b = per_user_a.align(per_user_b, join="inner")
    diff = (b - a).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_users": int(len(diff))}
