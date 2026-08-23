"""
recsys/fourstage_recsys/retrieval/two_tower.py
==============================================
Two-tower similarity learning for retrieval (Chapter 5).

Contains the training-pair construction (Listings 5.1-5.2), the baseline
two-tower model trained with binary cross-entropy (Listing 5.3), hard
negative mining (Section 5.3.2), the InfoNCE contrastive loss (Listing 5.7),
and the combined ``TwoTowerWithInfoNCE`` model.

Everything in this module works in **dense index space** (see
``recsys.data.preprocessing``); the ANN retrieval layer maps indices back to
string item IDs.

Implementation notes that deliberately differ from the naive versions:

  - ``TwoTower.forward`` returns **logits**, not probabilities, because the
    loss is ``BCEWithLogitsLoss`` (which applies the sigmoid internally).
    Applying a sigmoid in ``forward`` as well would squash every prediction
    into roughly (0.5, 0.73) and cripple the gradients. Use ``predict()``
    for probabilities at inference time.
  - ``hard_negative_mining`` samples from a **band of ranks**
    (``pool_start``-``pool_end``) rather than the strict top-k. The very top
    of the similarity ranking is where false negatives concentrate -- items
    the user would have loved but happened not to interact with. Mining the
    strict top-k punishes the model for its best predictions; the loss
    plateaus at chance level (ln(1 + num_negatives)) and the embedding space
    collapses toward uniformity.
  - ``train_infonce`` warms up on random negatives before mining begins, so
    mining operates on an embedding space that already has broad structure.

Usage
-----
from recsys.fourstage_recsys.retrieval.two_tower import (
    TwoTower, TwoTowerWithInfoNCE, InfoNCELoss,
    create_positive_pairs, create_negative_pairs,
    hard_negative_mining, train_bce, train_infonce, extract_embeddings,
)
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Training pairs (Listings 5.1 and 5.2)
# ---------------------------------------------------------------------------

def create_pairs_for_user(item_indices: list, max_pairs: int = 50) -> list:
    """Every unique co-consumption pair in one user's history, capped.

    The cap matters: a user with 200 rated movies produces 19,900 pairs, and
    a handful of very active users would otherwise dominate the training set.
    """
    pairs = list(itertools.combinations(item_indices, 2))    #A
    if len(pairs) > max_pairs:                              #B
        pairs = random.sample(pairs, max_pairs)
    return pairs

#A Every unique pair of items in the user's training history
#B Cap the quadratic blowup for very active users


def create_positive_pairs(train_items_by_user: dict,
                          max_pairs_per_user: int = 50) -> list:
    """Positive pairs across all users, in index space."""
    positive_pairs = []
    for items in train_items_by_user.values():
        positive_pairs.extend(create_pairs_for_user(items, max_pairs_per_user))
    return positive_pairs


def create_negative_pairs(positive_pairs: list, num_items: int,
                          num_neg_per_pos: int = 5) -> list:
    """Random negatives: pair the anchor of each positive with catalog samples.

    Both members of every pair live in index space -- the random draw is from
    ``range(num_items)``, never from raw item IDs.
    """
    neg_pairs = []
    for item1, item2 in positive_pairs:                     #A
        for _ in range(num_neg_per_pos):
            item3 = random.randrange(num_items)             #B
            while item3 == item1 or item3 == item2:         #C
                item3 = random.randrange(num_items)
            neg_pairs.append((item1, item3))
    return neg_pairs

#A Iterate through every positive pair
#B Sample a random item index from the full catalog
#C Ensure the sample is not part of the original positive pair


# ---------------------------------------------------------------------------
# The baseline two-tower model (Listing 5.3)
# ---------------------------------------------------------------------------

class TwoTower(nn.Module):
    """Two-tower model: independent encoders meeting at a dot product."""

    def __init__(self, num_items: int, emb_dim: int = 64,
                 c_vector: float = 1e-6):
        super().__init__()
        self.embedding1 = nn.Embedding(num_items, emb_dim)  #A
        self.embedding2 = nn.Embedding(num_items, emb_dim)  #A
        self.tower_one = self._build_tower(emb_dim)         #B
        self.tower_two = self._build_tower(emb_dim)         #B
        self.bce = nn.BCEWithLogitsLoss()                   #C
        self.c_vector = c_vector

    @staticmethod
    def _build_tower(emb_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, emb_dim))

    def forward(self, item_1: torch.Tensor, item_2: torch.Tensor) -> torch.Tensor:
        emb1 = self.tower_one(self.embedding1(item_1))      #D
        emb2 = self.tower_two(self.embedding2(item_2))      #D
        return torch.sum(emb1 * emb2, dim=1)                #E

    def predict(self, item_1: torch.Tensor, item_2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(item_1, item_2))  #F

    def query_table(self) -> torch.Tensor:
        all_idx = torch.arange(self.embedding1.num_embeddings,
                               device=self.embedding1.weight.device)
        return self.tower_one(self.embedding1(all_idx))

    def candidate_table(self) -> torch.Tensor:
        all_idx = torch.arange(self.embedding2.num_embeddings,
                               device=self.embedding2.weight.device)
        return self.tower_two(self.embedding2(all_idx))

    def loss(self, pred_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred_logits, label)             #G
        if not self.c_vector:
            return bce_loss
        reg = (self.embedding1.weight.square().sum()
               + self.embedding2.weight.square().sum()) * self.c_vector
        return bce_loss + reg                               #H

#A Two separate embedding layers, one per tower
#B Each tower is a small feed-forward network
#C BCEWithLogitsLoss applies the sigmoid internally -- forward must return logits
#D Pass each item through its embedding layer and tower
#E The dot product is the affinity logit
#F Probabilities only at inference time
#G Binary cross-entropy between logit and label
#H L2 regularization keeps embedding weights from growing without bound


# ---------------------------------------------------------------------------
# Hard negative mining (Section 5.3.2)
# ---------------------------------------------------------------------------

def hard_negative_mining(query_embeddings: torch.Tensor,
                         item_embeddings: torch.Tensor,
                         positive_indices: torch.Tensor,
                         query_indices: torch.Tensor,
                         num_hard_negatives: int = 2,
                         num_random_negatives: int = 6,
                         pool_start: int = 30,
                         pool_end: int = 150) -> torch.Tensor:
    """Mine hard negatives from a band of ranks, mixed with random negatives.

    See the module docstring for why the band (skipping the very top ranks)
    is essential: strict top-k mining concentrates false negatives and can
    collapse training entirely.
    """
    similarities = query_embeddings @ item_embeddings.T          #A
    batch = torch.arange(query_embeddings.shape[0],
                         device=similarities.device)
    similarities[batch, positive_indices] = -float("inf")        #B
    similarities[batch, query_indices] = -float("inf")           #B
    k = min(pool_end, similarities.shape[1] - 2)
    _, ranked = torch.topk(similarities, k=k, dim=1)
    pool = ranked[:, min(pool_start, k - 1):]                    #C
    pick = torch.randint(0, pool.shape[1],
                         (query_embeddings.shape[0], num_hard_negatives),
                         device=similarities.device)
    hard = pool.gather(1, pick)                                  #D
    rand = torch.randint(0, item_embeddings.shape[0],
                         (query_embeddings.shape[0], num_random_negatives),
                         device=similarities.device)             #E
    return torch.cat([hard, rand], dim=1)                        #F

#A Similarity between every query in the batch and every catalog item
#B Mask the positive AND the query item so neither is selected as a negative
#C The mining pool: challenging ranks, skipping the very top where false negatives concentrate
#D Sample hard negatives from the pool rather than taking the strict top-k
#E Random negatives preserve broad structure and training stability
#F The combined negative set for this batch


# ---------------------------------------------------------------------------
# InfoNCE (Listing 5.7)
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """Contrastive loss: identify the positive among the candidates."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> torch.Tensor:
        u = F.normalize(query_embeddings, dim=-1)                #A
        v_pos = F.normalize(positive_embeddings, dim=-1)
        v_neg = F.normalize(negative_embeddings, dim=-1)
        pos_scores = (u * v_pos).sum(dim=-1) / self.temperature  #B
        neg_scores = torch.bmm(
            u.unsqueeze(1), v_neg.transpose(1, 2)
        ).squeeze(1) / self.temperature                          #C
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  #D
        targets = torch.zeros(u.shape[0], dtype=torch.long,
                              device=u.device)                   #E
        return F.cross_entropy(logits, targets)                  #F

#A Normalize so the dot product equals cosine similarity
#B One similarity score for the positive
#C Similarity scores for every negative in the batch
#D Positive at index 0, negatives after it
#E The "correct class" is always index 0
#F Cross-entropy identifies the positive among all candidates


class TwoTowerWithInfoNCE(TwoTower):
    """Same architecture as TwoTower -- only the objective changes."""

    def __init__(self, num_items: int, emb_dim: int = 64,
                 temperature: float = 0.07,
                 num_hard: int = 2, num_rand: int = 6):
        super().__init__(num_items, emb_dim)
        self.infonce = InfoNCELoss(temperature)
        self.num_hard = num_hard
        self.num_rand = num_rand

    def candidate_table(self) -> torch.Tensor:
        all_idx = torch.arange(self.embedding2.num_embeddings,
                               device=self.embedding2.weight.device)
        return self.tower_two(self.embedding2(all_idx))          #A

    def training_step(self, q_idx: torch.Tensor, pos_idx: torch.Tensor,
                      mining_table: torch.Tensor | None = None) -> torch.Tensor:
        q = self.tower_one(self.embedding1(q_idx))               #B
        with torch.no_grad():
            if mining_table is not None:
                neg_idx = hard_negative_mining(
                    F.normalize(q.detach(), dim=-1),
                    F.normalize(mining_table, dim=-1),
                    pos_idx, q_idx, self.num_hard, self.num_rand)  #C
            else:
                neg_idx = torch.randint(                         #D
                    0, self.embedding2.num_embeddings,
                    (q_idx.shape[0], self.num_hard + self.num_rand),
                    device=q_idx.device)
        pos_emb = self.tower_two(self.embedding2(pos_idx))       #E
        neg_emb = self.tower_two(self.embedding2(neg_idx))       #E
        return self.infonce(q, pos_emb, neg_emb)                 #F

#B Encode the batch of query items
#C Hard negatives: mine from a pre-built table (no gradients needed for selection)
#D Warmup: random negatives until the space has broad structure
#E Compute gradients only for selected items, not the whole catalog
#F InfoNCE over the positive and the mined negatives


# ---------------------------------------------------------------------------
# Training loops and embedding extraction
# ---------------------------------------------------------------------------

def train_bce(model: TwoTower, positive_pairs: list, negative_pairs: list,
              epochs: int = 4, batch_size: int = 2048, lr: float = 1e-3,
              device: str = "cpu", verbose: bool = True):
    """Train the baseline two-tower model with BCE on labeled pairs.

    Returns
    -------
    (model, losses) : the trained model and a list of per-epoch average losses.
    """
    pairs = torch.cat([torch.tensor(positive_pairs, dtype=torch.long),
                       torch.tensor(negative_pairs, dtype=torch.long)])
    labels = torch.cat([torch.ones(len(positive_pairs)),
                        torch.zeros(len(negative_pairs))])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    pairs = pairs.to(device)
    labels = labels.to(device)
    n = len(pairs)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            i1 = pairs[idx, 0]
            i2 = pairs[idx, 1]
            y = labels[idx]
            optimizer.zero_grad()
            logits = model(i1, i2)
            loss = model.loss(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        epoch_loss = total / n
        losses.append(epoch_loss)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} -- loss {epoch_loss:.4f}")
    return model, losses


def train_infonce(model: TwoTowerWithInfoNCE, positive_pairs: list,
                  epochs: int = 10, batch_size: int = 1024, lr: float = 5e-3,
                  warmup_epochs: int = 2, device: str = "cpu",
                  verbose: bool = True):
    """Train with InfoNCE: random negatives during warmup, then mined ones.

    Returns
    -------
    (model, losses) : the trained model and a list of per-epoch average losses.
    """
    pairs = torch.tensor(positive_pairs, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    n = len(pairs)
    losses = []
    for epoch in range(epochs):
        if epoch >= warmup_epochs:                               #A
            model.eval()
            with torch.no_grad():
                mining_table = model.candidate_table()
            model.train()
        else:
            mining_table = None                                  #A
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            q = pairs[idx, 0]
            p = pairs[idx, 1]
            optimizer.zero_grad()
            loss = model.training_step(q, p, mining_table)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        epoch_loss = total / n
        losses.append(epoch_loss)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} -- loss {epoch_loss:.4f}")
    return model, losses

#A Build the mining table once per epoch after warmup; None signals random negatives


@torch.no_grad()
def extract_embeddings(model: TwoTower, device: str = "cpu"):
    """Extract L2-normalized query- and candidate-tower embedding tables.

    Returns
    -------
    (query, candidates) : two float32 arrays of shape (num_items, emb_dim).
    The candidate table is what goes into the ANN index; the query table
    encodes seed items at request time. Both are normalized so the inner
    product equals cosine similarity -- apply the same normalization
    everywhere or retrieval quality degrades silently.
    """
    model.eval()
    query = F.normalize(model.query_table(), dim=-1).cpu().numpy()
    cand = F.normalize(model.candidate_table(), dim=-1).cpu().numpy()
    return query.astype(np.float32), cand.astype(np.float32)
