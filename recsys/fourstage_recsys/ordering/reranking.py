"""List reranking: the heuristic half of the ordering stage (section 7.7).

These transforms reshape a relevance-ranked list AWAY from pure
relevance -- the opposite direction from chapter 5's candidate
reranking. Winning here is not a higher NDCG (that would mean the
reranker isn't doing its job); it is moving intra-list diversity while
relevance holds in band. Do not report offline NDCG lifts from these
transforms -- measuring their real effect is chapter 12's job.

Index conventions:
- mmr_rerank's sim_matrix is LOCAL to the candidate list: an NxN matrix
  over cand_ids, indexed by position (build it with candidate_similarity).
- apply_category_cap and order_stage work on raw movieIds; primary_genre
  maps movieId -> genre label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import ild_of_list, list_divergence
from .features import GENRES, ITEM_GENRE_COLS


def normalize_relevance(relevance) -> np.ndarray:
    """Min-max scale a list's scores to [0, 1].

    Ranker scores are unbounded and differ in scale from user to user,
    while similarity lives in [0, 1]. Without this, the same lambda means
    a different trade-off for every list.
    """
    r = np.asarray(relevance, dtype=float)
    span = r.max() - r.min()
    return (r - r.min()) / span if span > 0 else np.ones_like(r)


def mmr_rerank(cand_ids, relevance, sim_matrix, k, lam=0.7):
    """Listing 7.7: greedy Maximal Marginal Relevance.

    lam=1.0 recovers the pure relevance order; lam=0.0 ignores relevance
    entirely. `relevance` is normalized to [0, 1] per list.
    """
    rel = normalize_relevance(relevance)
    n = len(cand_ids)
    redundancy = np.zeros(n)                        # max sim to anything chosen so far
    chosen = np.zeros(n, dtype=bool)
    selected = []
    while len(selected) < min(k, n):
        score = lam * rel - (1 - lam) * redundancy
        score[chosen] = -np.inf
        best = int(np.argmax(score))
        selected.append(best)
        chosen[best] = True
        redundancy = np.maximum(redundancy, sim_matrix[:, best])  # O(n) update
    return [cand_ids[i] for i in selected]


def apply_category_cap(ranked_ids, primary_genre, k, cap=3):
    """Listing 7.9: at most `cap` items per genre in the final k."""
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
    """Listing 7.10: MMR reshapes the whole set, then the cap composes the final k."""
    reranked = mmr_rerank(cand_ids, relevance, sim_matrix, k=len(cand_ids), lam=lam)
    return apply_category_cap(reranked, primary_genre, k=k, cap=cap)


def candidate_similarity(cand_movie_ids, genre_matrix: pd.DataFrame) -> np.ndarray:
    """Cosine similarity over genre indicators, local to one candidate list."""
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
    """movieId -> one genre label for the cap: the movie's RAREST genre.

    MovieLens lists genres alphabetically, so "first genre" would label
    every action film Action and every adventure film Adventure -- the
    cap would then mostly be capping the alphabet. The rarest genre is
    the most distinctive one: Toy Story is Animation, not Adventure.
    """
    G = genre_matrix.set_index("movieId")[ITEM_GENRE_COLS]
    freq = G.sum(axis=0).to_numpy()
    labels = {}
    for movie_id, row in zip(G.index, G.to_numpy()):
        hits = np.flatnonzero(row > 0)
        labels[movie_id] = GENRES[hits[np.argmin(freq[hits])]] if hits.size else "None"
    return labels


def evaluate_ordering_pass(
    frame: pd.DataFrame, score_col: str, genre_matrix: pd.DataFrame,
    n_users: int = 500, k: int = 10, lam: float = 0.7, cap: int = 3, seed: int = 7,
) -> pd.DataFrame:
    """Table 7.4: ranked order vs +MMR vs +MMR+cap, averaged over a user sample.

    Reports only what this stage can honestly self-measure: relevance
    retained (share of the ranked top-k's normalized relevance the list
    keeps), ILD@k, max items per genre, and changed@k against the ranked order.
    """
    genres = primary_genre_map(genre_matrix)
    users = frame["userId"].drop_duplicates().sample(
        min(n_users, frame["userId"].nunique()), random_state=seed)
    acc = {"Ranked order": [], f"+MMR ({lam})": [], f"+MMR ({lam}) + genre cap ({cap})": []}
    for u in users:
        cand = frame[frame["userId"] == u].sort_values(score_col, ascending=False)
        if len(cand) < k:
            continue
        ids = cand["movieId"].tolist()
        rel = normalize_relevance(cand[score_col].to_numpy())
        sim = candidate_similarity(ids, genre_matrix)
        lists = {
            "Ranked order": ids[:k],
            f"+MMR ({lam})": mmr_rerank(ids, rel, sim, k=k, lam=lam),
            f"+MMR ({lam}) + genre cap ({cap})": order_stage(ids, rel, sim, genres,
                                                            k=k, lam=lam, cap=cap),
        }
        top_rel = rel[:k].sum()
        pos = {m: i for i, m in enumerate(ids)}
        for name, lst in lists.items():
            acc[name].append({
                "relevance retained": rel[[pos[m] for m in lst]].sum() / top_rel,
                f"ILD@{k}": ild_of_list(lst, ids, sim),
                "max items per genre": pd.Series([genres[m] for m in lst]).value_counts().max(),
                f"changed@{k} vs. ranked": list_divergence(np.array(ids), np.array(lst), k)[f"changed@{k}"],
            })
    return pd.DataFrame({name: pd.DataFrame(rows).mean() for name, rows in acc.items()}).T


def lambda_sweep(
    frame: pd.DataFrame, score_col: str, genre_matrix: pd.DataFrame,
    lams=np.linspace(0.0, 1.0, 11), n_users: int = 500, k: int = 10, seed: int = 7,
) -> pd.DataFrame:
    """Figure 7.5: ILD@k and relevance retained as lambda moves, averaged over users."""
    users = frame["userId"].drop_duplicates().sample(
        min(n_users, frame["userId"].nunique()), random_state=seed)
    rows = []
    for u in users:
        cand = frame[frame["userId"] == u].sort_values(score_col, ascending=False)
        if len(cand) < k:
            continue
        ids = cand["movieId"].tolist()
        rel = normalize_relevance(cand[score_col].to_numpy())
        sim = candidate_similarity(ids, genre_matrix)
        pos = {m: i for i, m in enumerate(ids)}
        for lam in lams:
            lst = mmr_rerank(ids, rel, sim, k=k, lam=lam)
            rows.append({"lambda": lam, f"ILD@{k}": ild_of_list(lst, ids, sim),
                         "relevance retained": rel[[pos[m] for m in lst]].sum() / rel[:k].sum()})
    return pd.DataFrame(rows).groupby("lambda").mean().reset_index()
