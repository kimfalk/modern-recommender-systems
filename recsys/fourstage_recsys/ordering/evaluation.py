"""Evaluation for the ordering stage (chapter 7).

The null hypothesis is the scored order: candidates sorted by their
chapter-5 cross-encoder score. Every model is measured against that row,
and list_divergence() reports how much the model actually changed the
list (overlap@k, changed@k, mean_shift@k) so a metric lift can't hide
behind a list that barely moved.

intra_list_diversity (Listing 7.8) lives here because it's a metric; the
list transforms it measures live in reranking.py.

NOTE: ndcg_at_k and mrr may duplicate the chapter-4 evaluation module in
the main repository. If so, delete the local copies and import from
there -- the signatures here are kept identical to the chapter listings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """NDCG@k for a relevance array already in ranked order."""
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.sum() <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(np.asarray(relevances, dtype=float))[::-1][:k]
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
    similarity matrix, pass positions (see ild_of_list for the id
    translation helper).
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
    """ILD for a list of raw ids, given the candidate-local sim matrix.

    `sim_matrix` is the NxN matrix over `cand_ids` (as built by
    reranking.candidate_similarity); this translates raw ids to
    positions before calling intra_list_diversity.
    """
    pos = {item: i for i, item in enumerate(cand_ids)}
    return intra_list_diversity([pos[i] for i in list_ids], sim_matrix)


class ScoredOrderBaseline:
    """The null hypothesis as a 'model': predicts the cross-encoder score.

    Feed it to evaluate_ranker to produce the baseline row of every
    result table. Requires cross_encoder_score in the feature columns.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["cross_encoder_score"].to_numpy()


def evaluate_ranker(test_frame, feature_cols, model, k: int = 10) -> dict:
    """Listing 7.3: score a ranker against the scored-order baseline.

    Keys are formatted with k; at the default k=10 they match the
    chapter tables (ndcg@10, mrr, overlap@10, changed@10, mean_shift@10).
    """
    ndcgs, mrrs, divs = [], [], []
    for _, cand in test_frame.groupby("userId"):
        scored_order = cand.sort_values("cross_encoder_score", ascending=False)
        preds = model.predict(cand[feature_cols])
        ranked_order = cand.assign(_s=preds).sort_values("_s", ascending=False)

        rel = ranked_order["relevant"].to_numpy()
        ndcgs.append(ndcg_at_k(rel, k))
        mrrs.append(mrr(rel))
        divs.append(list_divergence(
            scored_order["movieId"].to_numpy(),
            ranked_order["movieId"].to_numpy(),
            k=k,
        ))
    out = {f"ndcg@{k}": float(np.mean(ndcgs)), "mrr": float(np.mean(mrrs))}
    for key in divs[0]:
        out[key] = float(np.mean([d[key] for d in divs]))
    return out
