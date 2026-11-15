"""
recsys/fourstage_recsys/retrieval/sequential.py
================================================
SASRec and BERT4Rec sequential retrieval models for Chapter 6.

Both models share the same Transformer backbone. They differ in:
  - Attention masking  (causal vs. bidirectional)
  - Training objective (next-item prediction vs. masked-item / Cloze)
  - Inference         (rightmost position vs. appended [MASK] token)

Usage
-----
from recsys.fourstage_recsys.retrieval.sequential import (
    SASRec, BERT4Rec, gbce_loss,
    SASRecTrainDataset, BERT4RecTrainDataset, SequentialEvalDataset,
)
from recsys.evaluation.sequential import evaluate   # NDCG@k / HR@k, ablations

model = SASRec(num_items=59047)
# or
model = BERT4Rec(num_items=59047)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared Transformer backbone
# ---------------------------------------------------------------------------

class _SequentialBase(nn.Module):
    """
    Shared backbone used by both SASRec and BERT4Rec.

    Parameters
    ----------
    num_items : int
        Catalog size (real items are indexed 1..num_items; 0 = padding).
    max_len : int
        Maximum sequence length. Histories are left-padded to this length.
    hidden_dim : int
        Embedding and Transformer hidden dimension.
    num_layers : int
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability applied to embeddings and inside attention layers.
    extra_tokens : int
        Number of special tokens beyond the padding token (e.g. BERT4Rec's
        [MASK] token). These are appended after the real item indices so that
        the embedding table has shape (num_items + 1 + extra_tokens, hidden_dim).
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 200,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        extra_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        vocab_size = num_items + 1 + extra_tokens   # 0=pad, 1..num_items=items, then specials
        self.item_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)  # A
        self.pos_emb  = nn.Embedding(max_len, hidden_dim)                    # B
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,                                                  # C
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.final_norm  = nn.LayerNorm(hidden_dim)

    def _encode(
        self,
        sequences: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the shared embedding + Transformer stack.

        Parameters
        ----------
        sequences : (batch, seq_len) long tensor
        attn_mask : (seq_len, seq_len) bool tensor, True = block this pair.
                    None means fully bidirectional attention.

        Returns
        -------
        (batch, seq_len, hidden_dim) float tensor
        """
        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        x = self.item_emb(sequences) + self.pos_emb(positions)
        x = self.input_dropout(x)
        padding_mask = (sequences == 0)                                       # D

        # E: A left-padded position has zero valid keys under causal masking —
        # softmax over an all-masked row produces NaN. Across stacked layers
        # that NaN leaks into every position's output (0 * NaN = NaN in IEEE
        # float). Loop layers by hand and sanitize padding positions after
        # each one to prevent that leak from ever starting.
        for layer in self.transformer.layers:
            x = layer(x, src_mask=attn_mask, src_key_padding_mask=padding_mask)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)               # E
        return self.final_norm(x)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward().")

# A  0 is the padding index; real items start at 1.
# B  Learned positional embeddings, one per absolute position in the window.
# C  Pre-norm layout (LayerNorm before attention) — more stable to train.
# D  True wherever the sequence is padding; passed as src_key_padding_mask.
# E  softplus(-0) = log(2), not zero — sanitize between layers, not before.


# ---------------------------------------------------------------------------
# SASRec
# ---------------------------------------------------------------------------

class SASRec(_SequentialBase):
    """
    Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
    with causal (left-to-right) attention masking.

    The user state is read from the hidden state at the rightmost position.
    Trained with gBCE loss (Petrov & Macdonald, 2023) via ``gbce_loss``.
    """

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : (batch, seq_len) — left-padded item index sequences.

        Returns
        -------
        (batch, seq_len, hidden_dim) — hidden states at every position.
        The user state for scoring is ``output[:, -1, :]``.
        """
        seq_len = sequences.shape[1]
        causal_mask = torch.triu(                                             # A
            torch.ones(seq_len, seq_len, device=sequences.device, dtype=torch.bool),
            diagonal=1,
        )
        return self._encode(sequences, attn_mask=causal_mask)

    def recommend(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Return top-k item indices for each user in the batch."""
        with torch.no_grad():
            hidden     = self.forward(sequences)
            user_state = hidden[:, -1, :]                                     # B
            scores     = user_state @ self.item_emb.weight[:self.num_items + 1].T
            scores[:, 0] = -float("inf")                                      # C
        return torch.topk(scores, k, dim=1).indices

# A  Causal mask: position i cannot attend to positions j > i.
# B  The user state is the hidden vector at the last (rightmost) position.
# C  Never recommend the padding token (index 0).


# ---------------------------------------------------------------------------
# BERT4Rec
# ---------------------------------------------------------------------------

MASK_TOKEN_OFFSET = 1   # [MASK] lives at index num_items + 1


class BERT4Rec(_SequentialBase):
    """
    Sequential Recommendation with Bidirectional Encoder Representations
    from Transformer (Sun et al., 2019).

    Key differences from SASRec
    ---------------------------
    - No causal mask: every position can attend to every other position.
    - Training uses a Cloze (masked-item) objective rather than next-item
      prediction. ``mask_sequence`` replaces some input tokens with a
      special [MASK] token; the training loop calls this method explicitly
      before each forward pass, so the data flow is visible in the loop
      rather than hidden inside a dataset class.
    - Inference: a single [MASK] token is appended to the real history;
      the hidden state at that position is the user representation for
      scoring. The model never sees a future it shouldn't know about,
      because there is nothing to the right of the appended [MASK].

    The [MASK] token occupies index ``num_items + 1`` in the embedding table.
    """

    def __init__(self, num_items: int, mask_prob: float = 0.2, **kwargs) -> None:
        super().__init__(num_items, extra_tokens=1, **kwargs)
        self.mask_prob  = mask_prob
        self.mask_token = num_items + MASK_TOKEN_OFFSET

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : (batch, seq_len) — sequences that may contain [MASK]
                    tokens. At training time these are produced by
                    ``mask_sequence``; at inference time the caller appends
                    a single [MASK] token via ``recommend``.

        Returns
        -------
        (batch, seq_len, hidden_dim) — hidden states at every position.
        Prediction targets are the positions that carry a [MASK] token.
        """
        return self._encode(sequences, attn_mask=None)                        # A

    def mask_sequence(
        self,
        sequences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random Cloze masking for training.

        Called explicitly by the training loop before each forward pass,
        so that the masking step is visible in the loop rather than
        hidden inside a dataset.

        Each non-padding position is independently replaced with the
        [MASK] token with probability ``self.mask_prob``. The original
        item index is kept as the prediction target. At least one position
        is always masked so the loss is never zero.

        Parameters
        ----------
        sequences : (batch, seq_len) — raw (unmasked) item sequences,
                    already left-padded to max_len.

        Returns
        -------
        masked_seqs : (batch, seq_len) — sequences with [MASK] substitutions.
        cloze_mask  : (batch, seq_len) bool — True at every masked position.
                      Pass this as ``pos_mask`` to ``gbce_loss``.
        """
        non_pad    = (sequences != 0)
        rand       = torch.rand_like(sequences, dtype=torch.float)
        cloze_mask = non_pad & (rand < self.mask_prob)                        # B

        # Guarantee at least one masked position per sequence so the loss
        # is never zero for a user with a non-empty history.
        empty_rows = cloze_mask.sum(dim=1) == 0                               # C
        if empty_rows.any():
            last_real = (non_pad.cumsum(dim=1) * non_pad).argmax(dim=1)
            cloze_mask[empty_rows, last_real[empty_rows]] = True              # C

        masked_seqs = sequences.clone()
        masked_seqs[cloze_mask] = self.mask_token                             # D
        return masked_seqs, cloze_mask

    def recommend(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Append a [MASK] token to each history and return top-k next items.

        ``mask_sequence`` is NOT called here — at inference time there are
        no positions to predict from context; instead a single [MASK] token
        is appended so the model can predict what comes next.
        """
        batch_size = sequences.shape[0]
        mask_col = torch.full(
            (batch_size, 1), self.mask_token,
            dtype=torch.long, device=sequences.device,
        )
        augmented = torch.cat([sequences, mask_col], dim=1)[:, -self.max_len:]  # E
        with torch.no_grad():
            hidden     = self.forward(augmented)
            user_state = hidden[:, -1, :]                                       # F
            scores     = user_state @ self.item_emb.weight[:self.num_items + 1].T
            scores[:, 0] = -float("inf")
        return torch.topk(scores, k, dim=1).indices

# A  attn_mask=None — every position attends to every other; no causal constraint.
# B  Only non-padding positions are eligible for masking.
# C  Force at least one masked position so the loss never receives an empty mask.
# D  Replace the selected positions with the [MASK] token in the input.
# E  Trim to max_len — if the history was already full, the leftmost item is dropped.
# F  The [MASK] token is at the rightmost position, so hidden[:, -1, :] is the
#    user state — the same indexing as SASRec, but for a different reason.


# ---------------------------------------------------------------------------
# Loss function (shared)
# ---------------------------------------------------------------------------

def gbce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    pos_mask: torch.Tensor,
    num_items: int,
    t: float = 0.75,
) -> torch.Tensor:
    """
    Generalised binary cross-entropy loss (Petrov & Macdonald, 2023).

    Corrects the overconfidence that vanilla BCE with sampled negatives
    produces when the negative sampling rate α = n_neg / (N−1) is small
    relative to the catalog size N.

    Parameters
    ----------
    pos_scores : (batch, seq_len)        scores for the true next item.
    neg_scores : (batch, seq_len, n_neg) scores for sampled negatives.
    pos_mask   : (batch, seq_len) bool   True at positions to compute loss.
                 For SASRec: every non-padding position.
                 For BERT4Rec: only positions where mask_sequence placed [MASK].
    num_items  : int                     catalog size (excluding special tokens).
    t          : float in [0, 1]         calibration strength.
                 t=0 recovers plain BCE; t=1 is fully calibrated.

    Returns
    -------
    Scalar loss averaged over positions where pos_mask is True.

    Notes
    -----
    A common bug is to zero out scores at unwanted positions *before*
    computing the loss. softplus(-0) = log(2) ≈ 0.693, not zero, so
    that approach adds an irreducible constant for every excluded position
    and the reported loss plateaus early. Zero the per-position loss
    *after* it is computed, and average over real positions only.
    """
    n_neg  = neg_scores.shape[-1]
    alpha  = n_neg / max(num_items - 1, 1)
    beta   = alpha * (t * (1.0 - 1.0 / alpha) + 1.0 / alpha)                 # A

    pos_loss = beta * F.softplus(-pos_scores)                                 # B
    neg_loss = F.softplus(neg_scores).sum(dim=-1)                             # C
    per_position = (pos_loss + neg_loss) / (n_neg + 1)

    per_position = per_position * pos_mask.float()                            # D
    return per_position.sum() / pos_mask.float().sum().clamp(min=1.0)         # D

# A  β interpolates between no correction (t=0, β=1) and full correction (t=1, β=α).
# B  Numerically stable form of −β · log σ(s⁺).
# C  Numerically stable form of Σ −log(1 − σ(s⁻)), summed over negatives.
# D  Exclude non-target positions after computing the loss; average over targets only.


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def truncate_and_pad(seq: list[int], max_len: int) -> list[int]:
    """Left-pad ``seq`` to ``max_len``, truncating from the left if needed."""
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class SASRecTrainDataset(Dataset):
    """
    Each example is one user's (input, target) pair built by shifting
    their chronological history left by one position.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        num_items: int,
        max_len: int = 200,
        n_neg: int = 256,
    ) -> None:
        self.sequences = [s for s in sequences if len(s) >= 2]
        self.num_items = num_items
        self.max_len   = max_len
        self.n_neg     = n_neg

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq        = self.sequences[idx]
        input_seq  = torch.tensor(truncate_and_pad(seq[:-1], self.max_len), dtype=torch.long)
        target_seq = torch.tensor(truncate_and_pad(seq[1:],  self.max_len), dtype=torch.long)
        neg_items  = torch.randint(1, self.num_items + 1, (self.max_len, self.n_neg))
        return input_seq, target_seq, neg_items


class BERT4RecTrainDataset(Dataset):
    """
    Raw (unmasked) sequences for BERT4Rec training.

    Masking is NOT applied here. The training loop calls
    ``model.mask_sequence`` explicitly on each batch so that the Cloze
    masking step is visible in the loop rather than hidden inside the
    dataset. This dataset's job is simply to supply the unmasked history
    and the negatives needed for the gBCE loss.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        num_items: int,
        max_len: int = 200,
        n_neg: int = 256,
    ) -> None:
        self.sequences = [s for s in sequences if len(s) >= 2]
        self.num_items = num_items
        self.max_len   = max_len
        self.n_neg     = n_neg

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq        = self.sequences[idx]
        raw_seq    = torch.tensor(truncate_and_pad(seq, self.max_len), dtype=torch.long)  # A
        neg_items  = torch.randint(1, self.num_items + 1, (self.max_len, self.n_neg))
        return raw_seq, neg_items

# A  The full unmasked sequence — masking happens in the training loop via
#    model.mask_sequence, not here.


class SequentialEvalDataset(Dataset):
    """One example per user: the context sequence and the single held-out target."""

    def __init__(
        self,
        input_seqs: list[list[int]],
        targets: list[int],
        max_len: int = 200,
    ) -> None:
        self.input_seqs = input_seqs
        self.targets    = targets
        self.max_len    = max_len

    def __len__(self) -> int:
        return len(self.input_seqs)

    def __getitem__(self, idx: int):
        seq    = torch.tensor(truncate_and_pad(self.input_seqs[idx], self.max_len), dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return seq, target
