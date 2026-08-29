"""LambdaMART ordering model (chapter 7, sections 7.4.2-7.4.3).

Trees trained with a listwise loss: LightGBM's lambdarank objective over
per-user groups. The group array is the one thing to get right -- rows
per user must be contiguous, one count per user, counts summing to the
number of rows. train_lambdamart enforces that by sorting and counting.

For the cross-feature ablation (7.6.1), call train_lambdamart twice:
once with features.FEATURE_COLS and once with FEATURE_COLS_NO_CROSS.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def train_lambdamart(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 600,
) -> lgb.LGBMRanker:
    """Listing 7.2: fit a LambdaMART ranker on per-user candidate groups."""
    train_frame = train_frame.sort_values("userId")
    group_sizes = train_frame.groupby("userId").size().to_numpy()

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        label_gain=[0, 1],
        importance_type="gain",
        verbose=-1,
    )
    ranker.fit(
        train_frame[feature_cols],
        train_frame["relevant"],
        group=group_sizes,
        eval_at=[10],
    )
    return ranker


def explain_ranker(model, sample: pd.DataFrame, feature_cols: list[str]):
    """Listing 7.4: SHAP values and beeswarm for the tree ranker.

    Call on a held-out sample, not training rows, so the plot shows what
    the model generalized rather than what it memorized.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[feature_cols])
    shap.summary_plot(shap_values, sample[feature_cols])
    return shap_values


def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Gain-based importances as a tidy frame, for quick sanity checks."""
    return (
        pd.DataFrame({
            "feature": feature_cols,
            "gain": model.feature_importances_,
        })
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
