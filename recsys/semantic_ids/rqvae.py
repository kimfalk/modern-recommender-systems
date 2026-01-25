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
    def __init__(self, input_dim, embed_dim, codebook_sizes=[8, 64, 512]):
        super().__init__()
        self.depth = len(codebook_sizes)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, embed_dim)
        )
        
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(size, embed_dim) for size in codebook_sizes
        ])

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self._init_encoder()
        
    def _init_encoder(self):
        """Initialize encoder to approximately preserve input scale."""
        with torch.no_grad():
            # First linear layer: small weights
            nn.init.xavier_uniform_(self.encoder[0].weight, gain=1.0)
            nn.init.zeros_(self.encoder[0].bias)
            
            # Second linear layer: small weights
            nn.init.xavier_uniform_(self.encoder[2].weight, gain=1.0)
            nn.init.zeros_(self.encoder[2].bias)
            
            # Decoder: similar
            nn.init.xavier_uniform_(self.decoder[0].weight, gain=0.5)
            nn.init.zeros_(self.decoder[0].bias)
            nn.init.xavier_uniform_(self.decoder[2].weight, gain=0.5)
            nn.init.zeros_(self.decoder[2].bias)
    
    def forward(self, x):
        z = self.encoder(x)
        # z = F.normalize(z, p=2, dim=1)
        quantized_sum = 0
        residual = z
        total_loss = 0
        all_indices = []
        
        for i in range(self.depth):
            z_q, loss, indices = self.quantizers[i](residual)
            residual = residual - z_q
            quantized_sum += z_q
            total_loss += loss
            all_indices.append(indices)
        
        reconstructed = self.decoder(quantized_sum)
        
        codes = torch.stack(all_indices, dim=1)
        
        return reconstructed, total_loss, codes
