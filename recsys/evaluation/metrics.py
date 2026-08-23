import numpy as np


def average_precision_at_k(recommended_items, relevant_items, k):
    """Average Precision@k for a single user."""
    top_k = recommended_items[:k]
    hits = 0
    sum_precision = 0.0
    
    for i, item in enumerate(top_k):
        if item in relevant_items:
            hits += 1
            sum_precision += hits / (i + 1)
    
    if hits == 0:
        return 0.0
    
    return sum_precision / min(len(relevant_items), k)

def mrr_at_k(recommended_items, relevant_items, k):
    """Reciprocal Rank@k for a single user."""
    top_k = recommended_items[:k]
    for i, item in enumerate(top_k):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(recommended_items, relevant_items, k):
    """NDCG@k for a single user."""
    top_k = recommended_items[:k]
    relevance = [1.0 if item in relevant_items else 0.0 for item in top_k]
    
    if sum(relevance) == 0:
        return 0.0
    
    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    
    # IDCG: ideal ranking has all relevant items at the top
    ideal_relevance = sorted(relevance, reverse=True)
    n_relevant = min(len(relevant_items), k)
    ideal_relevance = [1.0] * n_relevant + [0.0] * (k - n_relevant)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def recall_at_k(recommended_items, relevant_items, k):
    """Recall@k for a single user."""
    if not relevant_items:
        return 0.0
    top_k = set(recommended_items[:k])
    return len(top_k & set(relevant_items)) / len(relevant_items)


def precision_at_k(recommended_items, relevant_items, k):
    """Precision@k for a single user."""
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / k


def catalog_coverage(all_recommended_items, num_items):
    """Fraction of the catalog that was recommended to at least one user.

    Low coverage at the retrieval stage is popularity bias showing up
    before ranking even starts.
    """
    return len(set(all_recommended_items)) / num_items
