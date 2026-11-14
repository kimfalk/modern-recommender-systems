"""Chapter 7 smoke test: the ordering stage end to end on synthetic
MovieLens-schema data, through the same code path as the real run.

chapter-5 preprocessing -> three-way split (checked against chapter 5's
split) -> upstream fit on 0-70% and 0-80% -> feature frames -> baselines,
LambdaMART, pointwise GBDT -> MMR + genre cap. Asserts structural
correctness; the synthetic numbers mean nothing. Run:
    python tests/ordering_smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from recsys.data.preprocessing import (
    add_item_idx, build_item_index, filter_min_item_ratings, filter_positive,
    sample_active_users, temporal_split_per_user,
)
from recsys.fourstage_recsys.ordering import (
    FEATURE_COLS, FEATURE_COLS_NO_CROSS, attach_labels, attach_upstream_scores,
    build_feature_frame, build_genre_matrix, evaluate_ordering_pass, evaluate_ranker,
    feature_history, make_synthetic_movielens, mmr_rerank, paired_bootstrap,
    per_user_temporal_slices, request_times, ScoredOrderBaseline, train_lambdamart,
    train_pointwise_gbdt, candidate_similarity,
)
from recsys.fourstage_recsys.ordering.upstream import fit_upstream, score_candidates

SMALL = dict(emb_dim=32, epochs=2, warmup_epochs=1, pool_start=10, pool_end=100,
             batch_size=256)


def main():
    torch.manual_seed(7)
    ratings, movies = make_synthetic_movielens(n_users=200, n_items=200, seed=7)
    ratings = filter_min_item_ratings(
        sample_active_users(ratings, n_users=10_000, min_ratings=20, seed=42), 10)
    pos = filter_positive(ratings, 4.0)
    item_ids, item_to_idx, _, _ = build_item_index(pos, movies)
    pos = add_item_idx(pos, item_to_idx)

    early, mid, late = per_user_temporal_slices(pos, cuts=(0.7, 0.8))
    tr, te = temporal_split_per_user(pos, test_frac=0.2)
    assert pd.concat([early, mid]).index.sort_values().equals(tr.index.sort_values())
    assert late.index.sort_values().equals(te.index.sort_values())

    up_fit = fit_upstream(early, item_ids, item_to_idx, infonce_params=SMALL,
                          cross_encoder_epochs=1)
    up_test = fit_upstream(pd.concat([early, mid]), item_ids, item_to_idx,
                           infonce_params=SMALL, cross_encoder_epochs=1)
    c_fit = score_candidates(up_fit, sorted(mid.userId.unique()), k_retrieve=50)
    c_test = score_candidates(up_test, sorted(late.userId.unique()), k_retrieve=50)

    # No candidate may be something the user already had in the upstream training slice
    seen = set(map(tuple, early[["userId", "movieId"]].to_numpy()))
    assert not any((u, m) in seen for u, m in c_fit[["userId", "movieId"]].to_numpy())

    genre_matrix = build_genre_matrix(movies)

    def frame(cands, labels, future):
        hist = feature_history(ratings, future)
        assert hist.merge(future[["userId", "movieId"]]).empty, "labels leaked into history"
        rows = attach_upstream_scores(attach_labels(cands[["userId", "movieId"]], labels), cands)
        return build_feature_frame(rows, hist, genre_matrix, request_ts=request_times(future))

    fit_frame = frame(c_fit, mid, pd.concat([mid, late]))
    test_frame = frame(c_test, late, late)
    assert not test_frame[FEATURE_COLS].isna().any().any(), "NaNs in feature frame"
    n_rel = late.groupby("userId").size().to_dict()

    base, pu_base = evaluate_ranker(test_frame, FEATURE_COLS, ScoredOrderBaseline(),
                                    n_relevant=n_rel, return_per_user=True)
    assert base["overlap@10"] == 1.0 and base["changed@10"] == 0

    results = {"scored order": base}
    for name, model, cols in [
        ("lambdamart", train_lambdamart(fit_frame, FEATURE_COLS, n_estimators=50), FEATURE_COLS),
        ("lambdamart no-x", train_lambdamart(fit_frame, FEATURE_COLS_NO_CROSS, n_estimators=50),
         FEATURE_COLS_NO_CROSS),
        ("gbdt pointwise", train_pointwise_gbdt(fit_frame, FEATURE_COLS, n_estimators=50),
         FEATURE_COLS),
    ]:
        res, pu = evaluate_ranker(test_frame, cols, model, n_relevant=n_rel,
                                  return_per_user=True)
        results[name] = res
        ci = paired_bootstrap(pu_base["ndcg@10"], pu["ndcg@10"], n_boot=200)
        assert ci["ci_low"] <= ci["diff"] <= ci["ci_high"]
    print(pd.DataFrame(results).T.round(4))

    # MMR: lam=1 is the ranked order; lam<1 never lowers ILD on average
    u = test_frame.userId.iloc[0]
    cand = test_frame[test_frame.userId == u].sort_values("cross_encoder_score", ascending=False)
    ids = cand.movieId.tolist()
    sim = candidate_similarity(ids, genre_matrix)
    assert mmr_rerank(ids, cand.cross_encoder_score.to_numpy(), sim, k=10, lam=1.0) == ids[:10]

    table = evaluate_ordering_pass(test_frame, "cross_encoder_score", genre_matrix, n_users=50)
    print(table.round(3))
    assert table.iloc[0]["relevance retained"] == 1.0
    assert table.iloc[2]["max items per genre"] <= 3
    assert table.iloc[1]["ILD@10"] >= table.iloc[0]["ILD@10"]
    print("chapter 7 smoke test passed")


if __name__ == "__main__":
    main()
