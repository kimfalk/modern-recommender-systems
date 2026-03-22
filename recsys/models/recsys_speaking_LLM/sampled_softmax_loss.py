import torch
import torch.nn as nn
import torch.nn.functional as F

class SampledSoftmaxLoss(nn.Module):
    def __init__(self, num_samples=10000):
        super().__init__()
        self.num_samples = num_samples
        
    def forward(self, hidden_states, target_ids, full_vocab_size):
        """
        Compute softmax over sampled vocabulary
        
        Args:
            hidden_states: [batch_size, hidden_dim]
            target_ids: [batch_size] ground truth items
            full_vocab_size: Total number of items
        """
        batch_size = hidden_states.size(0)
        
        # Always include target items
        sampled_ids = target_ids.clone()
        
        # Add random negative samples
        num_negatives = self.num_samples - len(target_ids)
        negatives = torch.randint(
            0, full_vocab_size, 
            (num_negatives,)
        )
        sampled_ids = torch.cat([sampled_ids, negatives])
        
        # Only compute logits for sampled items
        sampled_weights = self.embedding.weight[sampled_ids]
        logits = torch.matmul(hidden_states, sampled_weights.T)
        
        # Standard cross-entropy on reduced vocabulary
        return F.cross_entropy(logits, target_positions)

# Typical reduction: 1M vocabulary → 10K samples = 100x speedup
# Solution 2: Compressed Projection Head
# Add a bottleneck layer before the final projection to reduce embedding dimensions. This technique is used in systems like EGA and MTGR:

class CompressedDecodingHead(nn.Module):
    def __init__(self, hidden_dim=1024, compressed_dim=128, vocab_size=1_000_000):
        super().__init__()
        # Project down to compressed dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, compressed_dim),
            nn.ReLU(),
        )
        # Final vocabulary projection (much cheaper)
        self.vocab_projection = nn.Linear(compressed_dim, vocab_size)
        
    def forward(self, hidden_states):
        compressed = self.projection(hidden_states)  # [batch, seq, 128]
        logits = self.vocab_projection(compressed)   # [batch, seq, vocab]
        return logits
