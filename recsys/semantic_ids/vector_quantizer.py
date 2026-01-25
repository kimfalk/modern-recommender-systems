import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, 
                 commitment_cost=1.0, #0.25, 
                 decay=0.99, 
                 epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.decay = decay
        self.epsilon = epsilon
        
        # Buffer for the Codebook (Start Random)
        self.register_buffer('embedding', 
                             torch.randn(num_embeddings, embedding_dim) * 0.01) # scaling factor to start out with small values similar to the original embeddings. 
        
        # Buffers for EMA tracking
        self.register_buffer('cluster_size', 
                             torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', 
                             self.embedding.clone())
        
    def _update_centroids(self, data):
        # Convert to numpy for Sklearn
        data_np = data.detach().cpu().numpy()
        
        if data_np.std() < 1e-6:
            print(f"    ⚠️  Low variance, using random init")
            centroids = torch.randn(
                self.num_embeddings,
                self.embedding_dim,
                device=data.device,
                dtype=data.dtype
            ) * data_np.std()
        else:
            # NORMALIZE for K-means (it works better on unit sphere)
            data_normalized = data_np / (np.linalg.norm(data_np, axis=1, keepdims=True) + 1e-8)
            
            # Run K-means on normalized data
            kmeans = KMeans(
                n_clusters=self.num_embeddings,
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(data_normalized)
            
            # Get normalized centroids
            centroids_normalized = kmeans.cluster_centers_
            
            # DENORMALIZE: Scale centroids back to original data scale
            original_norms = np.linalg.norm(data_np, axis=1)
            avg_norm = original_norms.mean()
            
            # Scale normalized centroids to match original data magnitude
            centroids = centroids_normalized * avg_norm
            
            centroids = torch.tensor(centroids, device=data.device, dtype=data.dtype)
            
            print(f"    K-means: {kmeans.n_iter_} iterations")
            print(f"    Centroids: mean={centroids.mean():.4f}, std={centroids.std():.4f}")
        
        self.embedding.data.copy_(centroids)
        self.cluster_size.data.fill_(1.0)
        self.embed_avg.data.copy_(centroids)

        return centroids
    
    def init_codebook(self, data):
        """
        Runs K-Means on the data and resets the Codebook + EMA Buffers.
        This ensures we start with 'Alive' codes.
        """
        
        centroids = self._update_centroids(data)
        
        # RESET WEIGHTS
        self.embedding.data.copy_(centroids)
        
        # RESET EMA BUFFERS (Crucial!)
        # If we don't reset these, the old 'random' history will zero out our new centroids
        self.cluster_size.data.fill_(1.0) # Start with count 1 to avoid div-by-zero
        self.embed_avg.data.copy_(centroids) 

    def forward(self, inputs):
        # input_shape = inputs.shape
        # flat_input = inputs.view(-1, self.embedding_dim)
        
        # # 1. Distances
        # distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
        #             + torch.sum(self.embedding**2, dim=1) 
        #             - 2 * torch.matmul(flat_input, self.embedding.t()))
            
        # # 2. Encoding
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)
        
        # # 3. Quantize (Using F.embedding)
        # quantized = F.embedding(encoding_indices.squeeze(1), self.embedding).view(input_shape)
        
        
        # if self.training: # 4. TRAINING ONLY: Update Codebook via EMA
        #     self.ema_update_codebook(encodings, flat_input)
            
        # # 5. Loss & STE
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # quantized = inputs + (quantized - inputs).detach()
        
        # return quantized, loss, encoding_indices

        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding ** 2, dim=1) -
            2 * torch.matmul(flat_input, self.embedding.t())
        )
        
        # Find nearest
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Quantize
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding).view(input_shape)
        
        # Update codebook
        if self.training:
            self._ema_update(encodings, flat_input)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # Encoder commitment
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # Codebook loss
        
        # Combined loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.squeeze(1)
    
    def _ema_update(self, encodings, flat_input):
        
        dw = torch.sum(encodings, dim=0)
        self.cluster_size.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
        dw_embed = torch.matmul(encodings.t(), flat_input)
        self.embed_avg.data.mul_(self.decay).add_(dw_embed, alpha=1 - self.decay)
        
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        
        self.embedding.data.copy_(embed_normalized)

        return encodings, 
    