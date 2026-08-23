"""
recsys/fourstage_recsys/scoring/cross_encoder.py
================================================
Cross-encoder scoring for the retrieve-and-rerank pipeline (Chapter 5,
Sections 5.3-5.4).

Contains the MLP cross-encoder (Listing 5.8) with its training helpers, the
``CrossEncoderScoring`` stage (Listing 5.12), the ``CosineScoring`` baseline
it is compared against, and the text-based ``TransformerCrossEncoder``
(Listings 5.9 and 5.13). The transformer classes import the ``transformers``
library lazily, so this module works without it installed.

Two refinements over the minimal chapter listings, both of which matter in
production:

  - **Explicit interaction features.** The MLP input is ``[u, i, u * i]`` --
    a plain MLP on the concatenation ``[u, i]`` struggles to even
    *represent* a multiplicative interaction like a dot product; handing it
    the element-wise product lets the network start from the bi-encoder's
    answer and learn corrections to it.
  - **In-distribution negatives.** Training negatives are sampled partly
    from each positive item's retrieval neighbors rather than the full
    catalog. At serving time the reranker only ever sees candidates the
    retrieval stage already liked; training it exclusively on random
    negatives teaches a distinction every candidate it will ever score has
    already passed.

Usage
-----
from recsys.fourstage_recsys.scoring.cross_encoder import (
    CrossEncoderReranker, CrossEncoderScoring, CosineScoring,
    build_cross_encoder_training, train_cross_encoder,
    TransformerCrossEncoder, TransformerScoring,
)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
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


def build_cross_encoder_training(train_items_by_user: dict,
                                 neighbor_table: np.ndarray,
                                 num_items: int,
                                 num_hard: int = 2, num_random: int = 2,
                                 seed: int = 42):
    """Training triples (user_id, item_idx, label) with mixed negatives.

    ``neighbor_table`` holds each item's retrieval neighbors (index space,
    shape (num_items, n_neighbors)) -- the distribution the reranker will
    actually see at serving time.
    """
    users, items, labels = [], [], []
    rng = np.random.default_rng(seed)
    n_neighbors = neighbor_table.shape[1]
    for user_id, pos_items in train_items_by_user.items():
        pos_set = set(pos_items)
        for item in pos_items:
            users.append(user_id); items.append(item); labels.append(1.0)
            for _ in range(num_hard):                        #A
                neg = int(neighbor_table[item][rng.integers(n_neighbors)])
                tries = 0
                while neg in pos_set and tries < 10:
                    neg = int(neighbor_table[item][rng.integers(n_neighbors)])
                    tries += 1
                if neg in pos_set:
                    neg = int(rng.integers(num_items))
                users.append(user_id); items.append(neg); labels.append(0.0)
            for _ in range(num_random):                      #B
                neg = int(rng.integers(num_items))
                while neg in pos_set:
                    neg = int(rng.integers(num_items))
                users.append(user_id); items.append(neg); labels.append(0.0)
    return users, items, labels

#A In-distribution negatives: items retrieval would surface, that the user did not choose
#B Random negatives keep the broad structure


def train_cross_encoder(model: CrossEncoderReranker,
                        user_features: Dict, item_features: torch.Tensor,
                        users: list, items: list, labels: list,
                        epochs: int = 5, batch_size: int = 4096,
                        lr: float = 1e-3, device: str = "cpu",
                        verbose: bool = True) -> CrossEncoderReranker:
    """Standard BCE training over the (user, item, label) triples."""
    user_feat_matrix = torch.stack([user_features[u] for u in users])
    item_feat_matrix = item_features[items]
    label_tensor = torch.tensor(labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.to(device).train()
    n = len(labels)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(user_feat_matrix[idx].to(device),
                           item_feat_matrix[idx].to(device))
            loss = bce(logits, label_tensor[idx].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} -- loss {total / n:.4f}")
    return model


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


class CosineScoring(Scorer):
    """The Chapter 2 baseline scorer: similarity to the user's mean embedding."""

    def __init__(self, user_features: Dict, item_features: torch.Tensor,
                 item_to_idx: dict):
        self.user_features = user_features
        self.item_features = torch.nn.functional.normalize(item_features, dim=-1)
        self.item_to_idx = item_to_idx

    @torch.no_grad()
    def score(self, candidates: List[ScoredItem],
              context: RecommendationContext) -> List[ScoredItem]:
        if not candidates or context.user_id not in self.user_features:
            return candidates
        u = torch.nn.functional.normalize(
            self.user_features[context.user_id], dim=-1)
        idxs = [self.item_to_idx[c.item_id] for c in candidates]
        scores = self.item_features[idxs] @ u
        for item, s in zip(candidates, scores.tolist()):
            item.scores["cosine"] = float(s)
        return candidates


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
