"""Ordering stage (chapter 7): learned rankers + list reranking.

This package replaces the earlier ranking/ folder -- the models here are
ordering-stage models (trained against ordering metrics, evaluated
against the scored order from chapter 5).
"""
from .features import (
    GENRES, ITEM_GENRE_COLS, USER_AFF_COLS, USER_STAT_COLS, ITEM_STAT_COLS,
    CROSS_COLS, SCORE_COLS, FEATURE_COLS, FEATURE_COLS_NO_CROSS,
    DCN_DENSE_COLS, build_genre_matrix, build_user_features,
    build_item_features, build_cross_features, attach_labels,
    attach_upstream_scores, build_feature_frame,
)
from .evaluation import (
    ndcg_at_k, mrr, list_divergence, intra_list_diversity, ild_of_list,
    ScoredOrderBaseline, evaluate_ranker,
)
from .lambdamart import train_lambdamart, explain_ranker, feature_importance
from .dataset import (
    make_movie_index, add_movie_index, fit_scaler, CandidateDataset,
    make_loaders,
)
from .dcn import CrossLayer, DCNv2, train_dcn, validate, DCNRanker
from .reranking import (
    mmr_rerank, apply_category_cap, order_stage, candidate_similarity,
    primary_genre_map,
)
from .synthetic import make_synthetic_dataset
