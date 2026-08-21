"""LambdaMART ranker via LightGBM's lambdarank objective (listing 7.3)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "lambdarank_truncation_level": 30,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


def train_lambdamart(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    groups_valid: np.ndarray,
    params: dict | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    log_every: int = 50,
) -> lgb.Booster:
    """Train a LambdaMART ranker with early stopping on validation NDCG@10.

    ``groups_*`` are per-user list sizes in row order -- the rows MUST be
    sorted by user (``make_ranking_dataset`` guarantees this).  LightGBM
    will not warn you if they are not; it will just train on nonsense.
    """
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train)
    valid_set = lgb.Dataset(
        X_valid, label=y_valid, group=groups_valid, reference=train_set
    )
    return lgb.train(
        merged,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(log_every),
        ],
    )
