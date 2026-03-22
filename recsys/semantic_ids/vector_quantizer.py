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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, decay=0.95, 
                 usage_loss_weight=0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = 1e-5
        self.usage_loss_weight = usage_loss_weight
        
        # Non-learnable codebook (updated by EMA, not gradients)
        self.register_buffer('embedding',
                           torch.randn(num_embeddings, embedding_dim) * 0.01)
        self.register_buffer('cluster_size',
                           torch.zeros(num_embeddings))
        self.register_buffer('embed_avg',
                           self.embedding.clone())
        
        # Track dead codes
        self.register_buffer('_code_usage_count',
                           torch.zeros(num_embeddings))
        
        # Track recent usage for balancing
        self.register_buffer('_usage_smoothed',
                           torch.ones(num_embeddings) / num_embeddings)
    
    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding ** 2, dim=1) -
            2 * torch.matmul(flat_input, self.embedding.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding).view(input_shape)
        
        if self.training:
            self._update_codebooks(encodings, flat_input)
            # Track code usage
            self._code_usage_count += encodings.sum(0)
            
            # Update smoothed usage statistics
            batch_usage = encodings.mean(0)
            self._usage_smoothed.mul_(0.99).add_(batch_usage, alpha=0.01)
        
        # Standard VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Add entropy-based usage balancing loss
        # This encourages uniform code usage by maximizing entropy of usage distribution
        usage_loss_val = 0.0
        if self.training and self.usage_loss_weight > 0:
            # Compute entropy of usage (higher = more balanced)
            usage_probs = self._usage_smoothed + 1e-10
            usage_probs = usage_probs / usage_probs.sum()
            entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum()
            max_entropy = torch.log(torch.tensor(self.num_embeddings, dtype=torch.float32, device=inputs.device))
            
            # ✅ FIX: Positive loss that decreases as entropy increases
            # When entropy is low (bad): usage_loss is high
            # When entropy = max (perfect uniform): usage_loss = 0
            entropy_ratio = entropy / max_entropy  # 0 to 1
            usage_loss_val = 1.0 - entropy_ratio  # 1 to 0 (lower is better)
            loss = loss + self.usage_loss_weight * usage_loss_val
            
        # DEBUG: Print loss components on first batch
        if self.training and torch.rand(1).item() < 0.001:  # 0.1% chance to print
            print(f"  [VQ Debug] e_loss={e_latent_loss.item():.4f}, q_loss={q_latent_loss.item():.4f}, "
                  f"usage={usage_loss_val if isinstance(usage_loss_val, float) else usage_loss_val.item():.4f}, "
                  f"total={loss.item():.4f}")
        
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.squeeze(1)

    def _update_codebooks(self, encodings, flat_input):
        dw = torch.sum(encodings, dim=0)
        self.cluster_size.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        dw_embed = torch.matmul(encodings.t(), flat_input)
        self.embed_avg.data.mul_(self.decay).add_(dw_embed, alpha=1 - self.decay)
        
        n = self.cluster_size.sum()
        cluster_size = ((self.cluster_size + self.epsilon) / 
                       (n + self.num_embeddings * self.epsilon) * n)
        
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.data.copy_(embed_normalized)
    
    def reset_dead_codes(self, threshold=1.0):
        """Reinitialize codes that haven't been used."""
        if not self.training:
            return 0
        
        dead_codes = self._code_usage_count < threshold
        num_dead = dead_codes.sum().item()
        
        if num_dead > 0:
            # Reinitialize dead codes with random values from active codes
            active_codes = ~dead_codes
            if active_codes.sum() > 0:
                # Sample from active codes and add noise
                active_embeddings = self.embedding[active_codes]
                for i in torch.where(dead_codes)[0]:
                    idx = torch.randint(0, active_embeddings.shape[0], (1,))
                    noise = torch.randn_like(active_embeddings[idx]) * 0.01
                    self.embedding[i] = active_embeddings[idx] + noise
                    self.cluster_size[i] = 1.0
                    self.embed_avg[i] = self.embedding[i]
        
        # Reset counter
        self._code_usage_count.zero_()
        return num_dead
    
    def init_codebook(self, data):
        """Initialize with K-means on raw data (no normalization)."""
        data_np = data.detach().cpu().numpy()
        
        if data_np.std() < 1e-6 or len(data_np) < self.num_embeddings:
            return
        
        # CRITICAL: Don't normalize! Use the expanded space from encoder
        kmeans = KMeans(n_clusters=self.num_embeddings, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(data_np)
        
        centroids = kmeans.cluster_centers_
        
        self.embedding.data.copy_(
            torch.tensor(centroids, device=data.device, dtype=data.dtype)
        )
        self.cluster_size.data.fill_(1.0)
        self.embed_avg.data.copy_(self.embedding)
        
        print(f"    K-means: {kmeans.n_iter_} iters")

# class VectorQuantizerEMA(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, 
#                  commitment_cost=0.5, #0.25, 
#                  decay=0.95, 
#                  epsilon=1e-5):
#         super().__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.commitment_cost = commitment_cost
        
#         self.decay = decay
#         self.epsilon = epsilon
        
#         # Buffer for the Codebook (Start Random)
#         self.register_buffer('embedding', 
#                              torch.randn(num_embeddings, embedding_dim) * 0.01) # scaling factor to start out with small values similar to the original embeddings. 
#         # Buffers for EMA tracking
#         self.register_buffer('cluster_size', 
#                              torch.zeros(num_embeddings))
#         self.register_buffer('embed_avg', 
#                              self.embedding.clone())
        
#     def _update_centroids(self, data):
#         # Convert to numpy for Sklearn
#         data_np = data.detach().cpu().numpy()
        
#         if data_np.std() < 1e-6:
#             print(f"    ⚠️  Low variance, using random init")
#             centroids = torch.randn(
#                 self.num_embeddings,
#                 self.embedding_dim,
#                 device=data.device,
#                 dtype=data.dtype
#             ) * data_np.std()
#         else:
#             # NORMALIZE for K-means (it works better on unit sphere)
#             data_normalized = data_np / (np.linalg.norm(data_np, axis=1, keepdims=True) + 1e-8)
            
#             # Run K-means on normalized data
#             kmeans = KMeans(
#                 n_clusters=self.num_embeddings,
#                 n_init=10,
#                 max_iter=300,
#                 random_state=42
#             )
#             kmeans.fit(data_normalized)
            
#             # Get normalized centroids
#             centroids_normalized = kmeans.cluster_centers_
            
#             # DENORMALIZE: Scale centroids back to original data scale
#             original_norms = np.linalg.norm(data_np, axis=1)
#             avg_norm = original_norms.mean()
            
#             # Scale normalized centroids to match original data magnitude
#             centroids = centroids_normalized * avg_norm
            
#             centroids = torch.tensor(centroids, device=data.device, dtype=data.dtype)
            
#             print(f"    K-means: {kmeans.n_iter_} iterations")
#             print(f"    Centroids: mean={centroids.mean():.4f}, std={centroids.std():.4f}")
        
#         self.embedding.data.copy_(centroids)
#         self.cluster_size.data.fill_(1.0)
#         self.embed_avg.data.copy_(centroids)

#         return centroids
    

#     def forward(self, inputs):
  
#         input_shape = inputs.shape
#         flat_input = inputs.view(-1, self.embedding_dim)
        
#         # Calculate distances
#         distances = (
#             torch.sum(flat_input ** 2, dim=1, keepdim=True) +
#             torch.sum(self.embedding ** 2, dim=1) -
#             2 * torch.matmul(flat_input, self.embedding.t())
#         )
        
#         # Find nearest
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
#         # Quantize
#         encodings = torch.zeros(
#             encoding_indices.shape[0],
#             self.num_embeddings,
#             device=inputs.device
#         )
#         encodings.scatter_(1, encoding_indices, 1)
#         quantized = torch.matmul(encodings, self.embedding).view(input_shape)
        
#         # Update codebook
#         if self.training:
#             self._update_codebooks(encodings, flat_input)
        
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # Encoder commitment
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())  # Codebook loss
        
#         # Combined loss
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
#         # Straight-through estimator
#         quantized = inputs + (quantized - inputs).detach()
        
#         return quantized, loss, encoding_indices.squeeze(1)
    

    