import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from recsys.semantic_ids.vector_quantizer import VectorQuantizerEMA

class RQVAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 codebook_sizes=[8, 64, 512],
                 usage_loss_weight=2.0):
        super().__init__()
        self.depth = len(codebook_sizes)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)  # Final layer norm for stability
        )

        # EMA quantizers with strong usage balancing
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(size, embed_dim, 
                               commitment_cost=0.5, 
                               decay=0.90, 
                               usage_loss_weight=usage_loss_weight)
            for size in codebook_sizes
        ])

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # Mirror encoder architecture
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),  # Output must match input dimension!
        )
        self._init_encoder()
        
    def _init_encoder(self):
        """Initialize with standard gain for natural clustering."""
        with torch.no_grad():
            # Use gain=sqrt(2) for ReLU (Kaiming init principle)
            for module in self.encoder:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    nn.init.zeros_(module.bias)
            
            # Decoder: standard xavier
            for module in self.decoder:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    def forward(self, x, return_usage_stats=False, variance_weight=1.0):
        z = self.encoder(x)
        
        # CRITICAL: Prevent encoder collapse
        variance_loss = 0
        if self.training and variance_weight > 0:
            encoder_std = z.std(dim=0).mean()  # Mean std across dimensions
            target_std = 1.0  # Higher target - BERT embeddings have std ~1.0-2.0
            variance_loss = variance_weight * (encoder_std - target_std) ** 2
        
        quantized_sum = 0
        residual = z
        total_loss = variance_loss
        all_indices = []
        usage_stats = []
        
        for i in range(self.depth):
            z_q, loss, indices = self.quantizers[i](residual)
            residual = residual - z_q
            quantized_sum += z_q
            total_loss += loss
            all_indices.append(indices)
            
            # Track usage for diversity
            if return_usage_stats:
                usage_stats.append(indices)
        
        # Add small penalty for large residuals (optional but helpful)
        if self.training:
            residual_loss = 0.01 * torch.mean(residual ** 2)
            total_loss += residual_loss
        
        reconstructed = self.decoder(quantized_sum)
        
        codes = torch.stack(all_indices, dim=1)
        
        if return_usage_stats:
            return reconstructed, total_loss, codes, usage_stats
        
        return reconstructed, total_loss, codes
