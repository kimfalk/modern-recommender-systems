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
