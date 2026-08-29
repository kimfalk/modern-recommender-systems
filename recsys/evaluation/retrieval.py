"""
recsys/evaluation/retrieval.py
==============================
Evaluation harnesses for the retrieval stage (Chapter 5, Section 5.5).

Retrieval failures are invisible in end-to-end metrics -- precision@10
cannot tell you whether a great item was never retrieved or retrieved but
ranked poorly. These harnesses evaluate the retrieval stage in isolation.

Protocol notes:

  - The seed item comes from the user's **training history** (their most
    recent training interaction), never from the relevance set. Seeding
    from a relevant item and then excluding it from the candidates would
    systematically deflate recall, since that item can never be a hit.
  - Training items are excluded from the candidates -- a recommender that
    resurfaces the user's own history is not retrieving anything.

Usage
-----
from recsys.evaluation.retrieval import (
    evaluate_embedding_space, evaluate_retrieval_stage,
    intra_list_diversity, stratified_retrieval_recall,
    segment_users_by_activity, segment_items_by_popularity,
)
"""

from __future__ import annotations

import numpy as np

from recsys.evaluation.metrics import (
    catalog_coverage, ndcg_at_k, precision_at_k, recall_at_k,
)


def evaluate_embedding_space(query_vecs: np.ndarray, cand_vecs: np.ndarray,
                             relevance_sets: dict, train_items_by_user: dict,
                             k_recall: int = 100, k_top: int = 10) -> dict:
    """Evaluate an embedding space directly, by exact search (index space).

    Used to compare embedding models (e.g. BCE vs. InfoNCE training) before
    any ANN index enters the picture, so index approximation error cannot
    muddy the comparison.
    """
    recalls, precisions, ndcgs = [], [], []
    for user_id, relevant in relevance_sets.items():
        history = train_items_by_user.get(user_id)
        if not history or not relevant:
            continue
        seed = history[-1]                                  #A
        scores = cand_vecs @ query_vecs[seed]
        scores[history] = -np.inf                           #B
        scores[seed] = -np.inf
        ranked = np.argsort(-scores)[:k_recall].tolist()
        recalls.append(recall_at_k(ranked, relevant, k_recall))
        precisions.append(precision_at_k(ranked, relevant, k_top))
        ndcgs.append(ndcg_at_k(ranked, relevant, k_top))
    return {
        f"Recall@{k_recall}": float(np.mean(recalls)),
        f"Precision@{k_top}": float(np.mean(precisions)),
        f"NDCG@{k_top}": float(np.mean(ndcgs)),
        "users": len(recalls),
    }

#A Seed I2I retrieval from the most recent training interaction
#B Never rank items the user has already seen in training


def evaluate_retrieval_stage(retrieval, relevance_sets: dict,
                             train_items_by_user: dict,
                             k: int = 100) -> dict:
    """Evaluate a pipeline retrieval stage (Listing 5.15), in item-ID space.

    ``retrieval`` is any object exposing
    ``retrieve_similar_items(seed_ids, k) -> List[ScoredItem]`` and a
    ``num_items`` attribute.
    """
    recalls = []
    all_retrieved = set()
    for user_id, relevant in relevance_sets.items():
        history = train_items_by_user.get(user_id)
        if not relevant or not history:
            continue
        seed = history[-1]                                  #A
        candidates = retrieval.retrieve_similar_items([seed], k=k)
        candidate_ids = [c.item_id for c in candidates]
        all_retrieved.update(candidate_ids)                 #B
        recalls.append(recall_at_k(candidate_ids, relevant, k))  #C
    return {
        f"retrieval_recall@{k}": float(np.mean(recalls)),
        "catalog_coverage": catalog_coverage(all_retrieved,
                                             retrieval.num_items),  #D
        "num_users_evaluated": len(recalls),
    }

#A Seed from the most recent training interaction -- never from the relevance set
#B Track every item retrieved for anyone, across all users
#C Fraction of the relevance set present in the candidate pool
#D Fraction of the catalog that is visible to at least one user


def intra_list_diversity(retrieval, relevance_sets: dict,
                         train_items_by_user: dict,
                         item_embeddings: np.ndarray, item_to_idx: dict,
                         k: int = 100) -> float:
    """Average pairwise cosine distance within each user's candidate pool.

    High ILD means retrieval surfaces varied candidates, giving the scoring
    model something to choose from; low ILD means the pool is one tight
    cluster and downstream variety is already lost. Same protocol as
    ``evaluate_retrieval_stage``: seeded from the most recent training item.
    """
    per_user = []
    for user_id, relevant in relevance_sets.items():
        history = train_items_by_user.get(user_id)
        if not relevant or not history:
            continue
        candidates = retrieval.retrieve_similar_items([history[-1]], k=k)
        idxs = [item_to_idx[c.item_id] for c in candidates
                if c.item_id in item_to_idx]
        if len(idxs) < 2:
            continue
        vecs = item_embeddings[idxs]
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        sims = vecs @ vecs.T                                #A
        n = len(idxs)
        mean_sim = (sims.sum() - n) / (n * (n - 1))         #B
        per_user.append(1.0 - mean_sim)                     #C
    return float(np.mean(per_user))

#A All pairwise cosine similarities within the candidate pool
#B Mean over off-diagonal entries only -- self-similarity is excluded
#C Cosine distance = 1 - cosine similarity


def segment_users_by_activity(train_items_by_user: dict,
                              quantiles: tuple = (1 / 3, 2 / 3)) -> dict:
    """Split users into light / medium / heavy by training-history length.

    Activity terciles by default. With timestamped data, a recency-based
    "dormant" segment is the natural fourth split -- the mechanics are the
    same: any mapping of segment name to a set of user IDs works.
    """
    counts = {u: len(items) for u, items in train_items_by_user.items()}
    lo, hi = np.quantile(list(counts.values()), quantiles)
    return {
        "light": {u for u, c in counts.items() if c <= lo},
        "medium": {u for u, c in counts.items() if lo < c <= hi},
        "heavy": {u for u, c in counts.items() if c > hi},
    }


def segment_items_by_popularity(item_counts: dict,
                                top_frac: float = 0.2) -> dict:
    """Split items into popular (top ``top_frac`` by interactions) and long tail."""
    ranked = sorted(item_counts, key=item_counts.get, reverse=True)
    cut = max(1, int(len(ranked) * top_frac))
    return {
        "popular": set(ranked[:cut]),
        "long_tail": set(ranked[cut:]),
    }


def stratified_retrieval_recall(retrieval, relevance_sets: dict,
                                train_items_by_user: dict,
                                user_segments: dict | None = None,
                                item_segments: dict | None = None,
                                k: int = 100) -> dict:
    """Retrieval recall@k broken down by user segment and item segment.

    Averages hide failure modes: a model with high average recall but
    near-zero recall for light users (or long-tail items) has a problem the
    overall number never shows.

    ``user_segments`` maps segment name to a set of user IDs; a user's
    recall counts toward every segment containing them. ``item_segments``
    maps segment name to a set of item IDs; per segment, recall is computed
    against only the relevant items inside that segment.
    """
    user_recalls: dict = {name: [] for name in (user_segments or {})}
    item_recalls: dict = {name: [] for name in (item_segments or {})}
    for user_id, relevant in relevance_sets.items():
        history = train_items_by_user.get(user_id)
        if not relevant or not history:
            continue
        candidates = retrieval.retrieve_similar_items([history[-1]], k=k)
        candidate_set = {c.item_id for c in candidates}
        recall = len(candidate_set & relevant) / len(relevant)
        for name, users in (user_segments or {}).items():
            if user_id in users:                            #A
                user_recalls[name].append(recall)
        for name, items in (item_segments or {}).items():
            seg_relevant = relevant & items                 #B
            if seg_relevant:
                item_recalls[name].append(
                    len(candidate_set & seg_relevant) / len(seg_relevant))
    result = {}
    for name, values in user_recalls.items():
        result[f"recall@{k} ({name} users)"] = float(np.mean(values))
    for name, values in item_recalls.items():
        result[f"recall@{k} ({name} items)"] = float(np.mean(values))
    return result

#A The user's overall recall is attributed to their segment
#B Per item segment, recall is measured against only the relevant items in that segment
