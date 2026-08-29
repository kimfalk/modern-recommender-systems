"""Chapter 7 smoke test: the whole ordering stage, end to end, on
synthetic MovieLens-schema data.

Covers: features -> LambdaMART (with/without crosses) -> DCN-v2 ->
evaluation against the scored-order baseline -> MMR + genre cap + ILD.
Deterministic; asserts structural correctness, prints the metric table
for eyeballing. Run:  python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from recsys.fourstage_recsys.ordering import (
    FEATURE_COLS, FEATURE_COLS_NO_CROSS, DCN_DENSE_COLS,
    build_genre_matrix, build_feature_frame,
    ScoredOrderBaseline, evaluate_ranker, list_divergence,
    ild_of_list, train_lambdamart, feature_importance,
    make_movie_index, add_movie_index, fit_scaler, make_loaders,
    DCNv2, train_dcn, DCNRanker,
    mmr_rerank, order_stage, candidate_similarity, primary_genre_map,
    make_synthetic_dataset,
)


def train_test_split_by_user(frame, test_frac=0.3, seed=7):
    users = np.array(sorted(frame["userId"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_test = max(int(len(users) * test_frac), 1)
    test_users = set(users[:n_test].tolist())
    return (
        frame[~frame["userId"].isin(test_users)].reset_index(drop=True),
        frame[frame["userId"].isin(test_users)].reset_index(drop=True),
    )


def main():
    torch.manual_seed(7)
    train, heldout, movies, candidates = make_synthetic_dataset(seed=7)
    genre_matrix = build_genre_matrix(movies)

    frame = build_feature_frame(candidates, train, genre_matrix)
    missing = [c for c in FEATURE_COLS if c not in frame.columns]
    assert not missing, f"missing feature columns: {missing}"
    assert not frame[FEATURE_COLS].isna().any().any(), "NaNs in feature frame"

    fit_frame, test_frame = train_test_split_by_user(frame)

    results = {}
    results["scored order (baseline)"] = evaluate_ranker(
        test_frame, FEATURE_COLS, ScoredOrderBaseline()
    )
    b = results["scored order (baseline)"]
    assert abs(b["overlap@10"] - 1.0) < 1e-9
    assert b["changed@10"] == 0 and b["mean_shift@10"] == 0.0

    lm = train_lambdamart(fit_frame, FEATURE_COLS, n_estimators=60)
    results["lambdamart"] = evaluate_ranker(test_frame, FEATURE_COLS, lm)
    lm_nc = train_lambdamart(fit_frame, FEATURE_COLS_NO_CROSS, n_estimators=60)
    results["lambdamart (no crosses)"] = evaluate_ranker(
        test_frame, FEATURE_COLS_NO_CROSS, lm_nc
    )
    top_feats = feature_importance(lm, FEATURE_COLS).head(5)

    movie_index = make_movie_index(frame["movieId"])
    fit_idx = add_movie_index(fit_frame, movie_index)
    test_idx = add_movie_index(test_frame, movie_index)
    dcn_fit, dcn_val = train_test_split_by_user(fit_idx, test_frac=0.2, seed=11)
    scaler = fit_scaler(dcn_fit, DCN_DENSE_COLS)
    loaders = make_loaders(dcn_fit, dcn_val, DCN_DENSE_COLS, scaler, batch_size=512)
    model = DCNv2(
        n_movies=len(movie_index), dense_dim=len(DCN_DENSE_COLS),
        emb_dim=8, n_cross=2, mlp_dims=(32,),
    )
    model = train_dcn(model, *loaders, epochs=3, lr=1e-3)
    dcn = DCNRanker(model, scaler, DCN_DENSE_COLS)
    results["dcn-v2"] = evaluate_ranker(
        test_idx, DCN_DENSE_COLS + ["movie_idx"], dcn
    )

    for name, r in results.items():
        assert 0.0 <= r["ndcg@10"] <= 1.0 and 0.0 <= r["mrr"] <= 1.0, name

    one_user = test_frame[test_frame["userId"] == test_frame["userId"].iloc[0]]
    ranked = one_user.assign(_s=lm.predict(one_user[FEATURE_COLS])) \
                     .sort_values("_s", ascending=False)
    cand_ids = ranked["movieId"].tolist()
    relevance = ranked["_s"].to_numpy()
    sim = candidate_similarity(cand_ids, genre_matrix)
    genres = primary_genre_map(genre_matrix)

    pure = mmr_rerank(cand_ids, relevance, sim, k=10, lam=1.0)
    assert pure == cand_ids[:10], "MMR with lam=1.0 must recover pure relevance order"

    final = order_stage(cand_ids, relevance, sim, genres, k=10, lam=0.6, cap=3)
    assert len(final) == 10 and len(set(final)) == 10
    counts = {}
    for m in final:
        counts[genres[m]] = counts.get(genres[m], 0) + 1
    assert max(counts.values()) <= 3, f"genre cap violated: {counts}"

    ild_before = ild_of_list(cand_ids[:10], cand_ids, sim)
    ild_after = ild_of_list(final, cand_ids, sim)
    div = list_divergence(np.array(cand_ids), np.array(final), k=10)

    print("\n=== chapter 7 smoke test: PASS ===\n")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nTop LambdaMART features by gain:")
    print(top_feats.to_string(index=False))
    print(f"\nOrdering pass on one user: ILD@10 {ild_before:.3f} -> {ild_after:.3f}, "
          f"changed@10 = {div['changed@10']}, overlap@10 = {div['overlap@10']:.2f}")


if __name__ == "__main__":
    main()
