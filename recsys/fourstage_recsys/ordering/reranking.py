"""List reranking: the heuristic half of the ordering stage (section 7.7).

These transforms reshape a relevance-ranked list AWAY from pure
relevance -- the opposite direction from chapter 5's candidate
reranking. Winning here is not a higher NDCG (that would mean the
reranker isn't doing its job); it is moving intra-list diversity while
relevance holds in band. Do not report offline NDCG lifts from these
transforms -- measuring their real effect is chapter 12's job.

Index conventions, to keep the chapter listings intact:
- mmr_rerank's sim_matrix is LOCAL to the candidate list: an NxN matrix
  over cand_ids, indexed by position (build it with candidate_similarity).
- apply_category_cap and order_stage work on raw movieIds; primary_genre
  maps movieId -> genre label.
- For ILD on the final list, use evaluation.ild_of_list, which
  translates raw ids back to positions in the local matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import GENRES, ITEM_GENRE_COLS


def mmr_rerank(cand_ids, relevance, sim_matrix, k, lam=0.7):
    """Listing 7.7: greedy Maximal Marginal Relevance.

    lam=1.0 recovers the pure relevance order; lam=0.0 ignores relevance
    entirely. sim_matrix is positional over cand_ids.
    """
    selected, remaining = [], list(range(len(cand_ids)))
    while len(selected) < k and remaining:
        best, best_score = None, -np.inf
        for r in remaining:
            redundancy = max((sim_matrix[r, s] for s in selected), default=0.0)
            score = lam * relevance[r] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = r, score
        selected.append(best)
        remaining.remove(best)
    return [cand_ids[i] for i in selected]


def apply_category_cap(ranked_ids, primary_genre, k, cap=3):
    """Listing 7.9: at most `cap` items per genre in the final k.

    This reshapes the scored set to compose a better list; it does not
    remove on eligibility -- that is filtering's job, before scoring.
    """
    counts, out = {}, []
    for item in ranked_ids:
        g = primary_genre[item]
        if counts.get(g, 0) >= cap:
            continue
        out.append(item)
        counts[g] = counts.get(g, 0) + 1
        if len(out) == k:
            break
    return out


def order_stage(cand_ids, relevance, sim_matrix, primary_genre,
                k=10, lam=0.7, cap=3):
    """Listing 7.10: the ordering stage end to end.

    MMR first reshapes for diversity across the whole candidate set,
    then the cap composes the final k.
    """
    reranked = mmr_rerank(cand_ids, relevance, sim_matrix,
                          k=len(cand_ids), lam=lam)
    capped = apply_category_cap(reranked, primary_genre, k=k, cap=cap)
    return capped


def candidate_similarity(cand_movie_ids, genre_matrix: pd.DataFrame) -> np.ndarray:
    """Cosine similarity over genre indicators, local to one candidate list.

    Returns the NxN positional matrix mmr_rerank and ild_of_list expect.
    """
    G = (
        genre_matrix.set_index("movieId")
        .loc[list(cand_movie_ids), ITEM_GENRE_COLS]
        .to_numpy(dtype=float)
    )
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Gn = G / norms
    return Gn @ Gn.T


def primary_genre_map(genre_matrix: pd.DataFrame) -> dict:
    """movieId -> a single genre label for the cap.

    Simplification noted in the chapter: MovieLens titles carry several
    genres; here we take the first indicator in canonical GENRES order
    (the repository can cap on full genre membership instead).
    """
    indicators = genre_matrix.set_index("movieId")[ITEM_GENRE_COLS].to_numpy()
    labels = {}
    for movie_id, row in zip(genre_matrix["movieId"].to_numpy(), indicators):
        hits = np.flatnonzero(row > 0)
        labels[int(movie_id)] = GENRES[hits[0]] if hits.size else "None"
    return labels
