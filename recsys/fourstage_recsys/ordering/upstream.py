"""Chapter-5 upstream stages, packaged for the ordering stage (chapter 7).

The ordering model needs, for every user it trains or is tested on, the
candidate list chapter 5 would have produced and the cross-encoder score
of each candidate. This module fits the chapter-5 retrieval + scoring
models on a given training slice and exports those scored candidates.

It is called twice (see splits.py): once on the 0-70% slice to produce
the ranker's training rows, and once on the 0-80% slice (chapter 5's own
training set) to produce its test rows. The pipeline mirrors
notebooks/chapter-05/03_full_pipeline.ipynb exactly: seed = the user's
most recent training item, ANN retrieval of `k_retrieve` candidates,
history filtering, cross-encoder scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from recsys.data.preprocessing import user_item_lists
from recsys.fourstage_recsys.filtering.history_filtering import HistoryFiltering
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.retrieval.ann_retrieval import (
    ANNRetrieval, ANNRetrievalIndex,
)
from recsys.fourstage_recsys.retrieval.two_tower import (
    TwoTowerWithInfoNCE, create_positive_pairs, extract_embeddings, train_infonce,
)
from recsys.fourstage_recsys.scoring.cross_encoder import (
    CrossEncoderReranker, CrossEncoderScoring,
    build_cross_encoder_training, train_cross_encoder,
)

# Chapter-5 hyperparameters (01_similarity_learning.ipynb, 03_full_pipeline.ipynb)
CH5_INFONCE_PARAMS = dict(
    emb_dim=128, num_hard=4, num_rand=6, temperature=0.07,
    pool_start=200, pool_end=1000,
    epochs=20, warmup_epochs=2, batch_size=1024, lr=5e-3,
)
CH5_CROSS_ENCODER_EPOCHS = 5


@dataclass
class UpstreamModels:
    item_ids: list
    item_to_idx: dict
    query_embeddings: np.ndarray
    item_embeddings: np.ndarray
    cross_encoder: torch.nn.Module
    user_features: dict
    ann_index: ANNRetrievalIndex
    train_items_ids: dict = field(default_factory=dict)


def fit_upstream(
    train_pos: pd.DataFrame,
    item_ids: list,
    item_to_idx: dict,
    embeddings: tuple[np.ndarray, np.ndarray] | None = None,
    infonce_params: dict | None = None,
    cross_encoder_epochs: int = CH5_CROSS_ENCODER_EPOCHS,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> UpstreamModels:
    """Fit chapter-5 retrieval (two-tower InfoNCE) and scoring (cross-encoder).

    `train_pos` must carry userId, movieId, item_idx, timestamp. Pass
    `embeddings=(query, cand)` to reuse two-tower tables already trained
    on exactly this slice (e.g. chapter 5's embeddings.npz for the 80% cut).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    p = {**CH5_INFONCE_PARAMS, **(infonce_params or {})}
    num_items = len(item_ids)
    train_items_idx = user_item_lists(train_pos, item_col="item_idx")

    if embeddings is None:
        pairs = create_positive_pairs(train_items_idx, max_pairs_per_user=50)
        model = TwoTowerWithInfoNCE(
            num_items, emb_dim=p["emb_dim"], temperature=p["temperature"],
            num_hard=p["num_hard"], num_rand=p["num_rand"],
            pool_start=p["pool_start"], pool_end=p["pool_end"],
        )
        model, _ = train_infonce(
            model, pairs, epochs=p["epochs"], warmup_epochs=p["warmup_epochs"],
            batch_size=p["batch_size"], lr=p["lr"], device=device, verbose=verbose,
        )
        query, cand = extract_embeddings(model, device=device)
    else:
        query, cand = embeddings
    emb_dim = cand.shape[1]

    ann_index = ANNRetrievalIndex(index_type="hnsw", emb_dim=emb_dim)
    ann_index.add_items(item_ids, cand)

    item_features = torch.tensor(cand)
    user_features = {
        u: torch.tensor(cand[idxs].mean(axis=0)) for u, idxs in train_items_idx.items()
    }
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    _, neighbor_table = ann_index.index.search(
        np.ascontiguousarray(qn.astype(np.float32)), 50)
    users, items, labels = build_cross_encoder_training(
        train_items_idx, neighbor_table, num_items, seed=seed)
    cross_encoder = train_cross_encoder(
        CrossEncoderReranker(emb_dim, emb_dim), user_features, item_features,
        users, items, labels, epochs=cross_encoder_epochs, device=device,
        verbose=verbose,
    )
    return UpstreamModels(
        item_ids=item_ids, item_to_idx=item_to_idx,
        query_embeddings=query, item_embeddings=cand,
        cross_encoder=cross_encoder.eval(), user_features=user_features,
        ann_index=ann_index,
        train_items_ids=user_item_lists(train_pos, item_col="movieId"),
    )


def score_candidates(
    upstream: UpstreamModels,
    users,
    k_retrieve: int = 100,
    device: str = "cpu",
) -> pd.DataFrame:
    """Retrieve -> filter -> score for each user; one row per candidate.

    Returns userId, movieId, retrieval_similarity, cross_encoder_score.
    Users with no training history (no seed) are skipped.
    """
    retrieval = ANNRetrieval(upstream.ann_index, upstream.query_embeddings,
                             upstream.item_to_idx)
    history = HistoryFiltering(upstream.train_items_ids)
    scorer = CrossEncoderScoring(
        upstream.cross_encoder, upstream.user_features,
        torch.tensor(upstream.item_embeddings), upstream.item_to_idx, device=device,
    )
    rows = []
    for user_id in users:
        seen = upstream.train_items_ids.get(user_id)
        if not seen:
            continue
        ctx = RecommendationContext(user_id=user_id, seed_items=[seen[-1]], k=k_retrieve)
        cands = retrieval.retrieve_similar_items(ctx.seed_items, k=k_retrieve)
        cands = scorer.score(history.filter(cands, ctx), ctx)
        rows.extend(
            (user_id, c.item_id, c.scores["similarity"], c.scores["cross_encoder"])
            for c in cands
        )
    return pd.DataFrame(rows, columns=[
        "userId", "movieId", "retrieval_similarity", "cross_encoder_score"])
