"""
recsys/fourstage_recsys/scoring/cross_encoder.py
================================================
Cross-encoder scoring for the retrieve-and-rerank pipeline (Chapter 5,
Section 5.6).

Contains the MLP cross-encoder (Listing 5.8), the ``CrossEncoderScoring``
stage that plugs it into the four-stage pipeline (Listing 5.12), and the
text-based ``TransformerCrossEncoder`` with its ``TransformerScoring`` stage
(Listings 5.9 and 5.13). The transformer classes import the ``transformers``
library lazily, so this module works without it installed.

The MLP input is ``[u, i, u * i]`` rather than the concatenation ``[u, i]``:
a plain MLP struggles to even *represent* a multiplicative interaction like a
dot product, so handing it the element-wise product lets the network start
from the bi-encoder's answer and learn corrections to it. User and item
features must therefore have the same dimension.

Training lives in ``cross_encoder_training``: time-split training rows built
from the candidates retrieval actually returns, and ``fit_cross_encoder``
with early stopping. ``recsys.evaluation.rescoring`` compares the rescored
order with retrieval order and popularity order.

Usage
-----
from recsys.fourstage_recsys.scoring.cross_encoder import (
    CrossEncoderReranker, CrossEncoderScoring,
    TransformerCrossEncoder, TransformerScoring,
)
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.stages.scoring import Scorer


# ---------------------------------------------------------------------------
# The MLP cross-encoder (Listing 5.8)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(nn.Module):
    """Joint encoder over concatenated user and item features."""

    def __init__(self, user_feature_dim: int, item_feature_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        assert user_feature_dim == item_feature_dim, \
            "interaction features assume matching dimensions"
        input_dim = user_feature_dim + 2 * item_feature_dim  #A
        self.network = nn.Sequential(                        #B
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_features: torch.Tensor,
                item_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([
            user_features, item_features,
            user_features * item_features,                   #C
        ], dim=-1)
        return self.network(combined).squeeze(-1)            #D

#A User features, item features, and their element-wise product
#B A small interaction network scores the joint input
#C Explicit interaction features -- see the module docstring
#D One relevance logit per user-item pair


# ---------------------------------------------------------------------------
# Scoring stages (Listing 5.12)
# ---------------------------------------------------------------------------

class CrossEncoderScoring(Scorer):
    """Scoring stage wrapping a trained cross-encoder.

    Scores all candidates in a single batched forward pass -- which is what
    you would do in production, and reduces latency substantially when
    scoring hundreds of candidates.
    """

    def __init__(self, cross_encoder: nn.Module,
                 user_features: Dict, item_features: torch.Tensor,
                 item_to_idx: dict, device: str = "cpu"):
        self.cross_encoder = cross_encoder                  #A
        self.user_features = user_features
        self.item_features = item_features
        self.item_to_idx = item_to_idx
        self.device = device

    @torch.no_grad()
    def score(self, candidates: List[ScoredItem],
              context: RecommendationContext) -> List[ScoredItem]:
        if not candidates or context.user_id not in self.user_features:
            return candidates
        idxs = [self.item_to_idx[c.item_id] for c in candidates]  #B
        user_feat = self.user_features[context.user_id]     #C
        item_feats = self.item_features[idxs]
        user_batch = user_feat.expand(len(candidates), -1)  #D
        scores = self.cross_encoder(
            user_batch.to(self.device), item_feats.to(self.device))  #E
        for item, s in zip(candidates, scores.cpu().tolist()):
            item.scores["cross_encoder"] = float(s)         #F
        return candidates

#A The trained cross-encoder
#B Map candidate item IDs into the model's index space
#C Precomputed user features, looked up per request
#D The user's features repeated across the batch
#E One batched forward pass scores every candidate
#F Scores are attached to the candidates and travel down the pipeline


# ---------------------------------------------------------------------------
# The transformer cross-encoder (Listings 5.9 and 5.13)
# ---------------------------------------------------------------------------

class TransformerCrossEncoder(nn.Module):
    """Text-based cross-encoder: user history and item description as text.

    Requires the ``transformers`` library and downloads the pretrained model
    on first use (~420 MB for bert-base-uncased).
    """

    def __init__(self, model_name: str = "bert-base-uncased",
                 hidden_dim: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer   # lazy import
        self.encoder = AutoModel.from_pretrained(model_name)      #A
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size
        self.scorer = nn.Sequential(                              #B
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_texts: List[str],
                item_texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            user_texts, item_texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.encoder.device)                                 #C
        outputs = self.encoder(**inputs)                          #D
        cls_embedding = outputs.last_hidden_state[:, 0, :]        #E
        return self.scorer(cls_embedding).squeeze(-1)             #F

#A A pretrained transformer is the encoder -- you rarely train one from scratch
#B A small scoring head on top of the [CLS] representation
#C The tokenizer joins user and item text with a [SEP] token, as BERT expects
#D Self-attention attends jointly across both inputs
#E The [CLS] token summarizes the joint encoding
#F One relevance logit per pair


class TransformerScoring(Scorer):
    """Scoring stage for the text-based cross-encoder."""

    def __init__(self, cross_encoder: TransformerCrossEncoder,
                 user_histories: Dict[str, str],
                 item_descriptions: Dict[str, str]):
        self.cross_encoder = cross_encoder
        self.user_histories = user_histories                      #A
        self.item_descriptions = item_descriptions

    @torch.no_grad()
    def score(self, candidates: List[ScoredItem],
              context: RecommendationContext) -> List[ScoredItem]:
        if not candidates or context.user_id not in self.user_histories:
            return candidates
        user_text = self.user_histories[context.user_id]
        item_texts = [self.item_descriptions[c.item_id]
                      for c in candidates]                        #B
        scores = self.cross_encoder(
            [user_text] * len(candidates), item_texts)            #C
        for item, s in zip(candidates, scores.cpu().tolist()):
            item.scores["cross_encoder"] = float(s)
        return candidates

#A User histories represented as text -- recent titles are a strong starting point
#B Collect the candidates' descriptions
#C Score every candidate in a single batched forward pass
