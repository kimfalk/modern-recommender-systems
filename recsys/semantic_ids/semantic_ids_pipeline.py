import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import mlflow
import mlflow.pytorch

from recsys.semantic_ids.rqvae import RQVAE
from recsys.semantic_ids.vector_quantizer import VectorQuantizerEMA

class SemanticIDPipeline:

    def __init__(self, 
                 codebook_sizes=[8, 32, 128], 
                 internal_dim=64,
                 usage_loss_weight=10.0):  # Strong penalty to force code sharing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading BERT...")
        # Using all-MiniLM-L6-v2 because it doesn't normalize to unit sphere
        # This avoids the clustering problems we had with e5-large-v2
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        bert_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        self.codebook_sizes = codebook_sizes
        self.rqvae = RQVAE(bert_dim, 
                           internal_dim, 
                           codebook_sizes,
                           usage_loss_weight=usage_loss_weight).to(self.device)
        # Lower learning rate for more stable training
        self.optimizer = optim.Adam(self.rqvae.parameters(), lr=5e-4)

    def initialize_data_with_embeddings(self, embeddings):
        
        data = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        print("\nInitializing codebooks...")
        print(f"Input embeddings: mean={data.mean():.4f}, std={data.std():.4f}")
        
        with torch.no_grad():
            latents = self.rqvae.encoder(data)
            
            print(f"Encoder output: mean={latents.mean():.4f}, std={latents.std():.4f}, shape={latents.shape}")
            
            # Check diversity
            pairwise_dist = torch.cdist(latents[:100], latents[:100]).mean()
            print(f"Avg pairwise distance (first 100): {pairwise_dist:.4f}")
            
            residual = latents
            
            for i in range(self.rqvae.depth):
                print(f"\n  Level {i+1}:")
                print(f"    Residual: mean={residual.mean():.4f}, std={residual.std():.4f}, min={residual.min():.4f}, max={residual.max():.4f}")
                
                # Simple K-means initialization
                self.rqvae.quantizers[i].init_codebook(residual)
                
                # Get residual for next level
                z_q, _, _ = self.rqvae.quantizers[i](residual)
                residual = residual - z_q
        
        return data

    def initialize_data(self, texts):
        print("\nEncoding texts...")
        embeddings = self.text_encoder.encode(texts, 
                                              show_progress_bar=True)
        
        return self.initialize_data_with_embeddings(embeddings)
    
    def train(self,
              data,
              epochs=100, 
              batch_size=64,
              diversity_weight=0.0):
            
        # 2. Train Loop
        print("Training RQ-VAE...")
        self.rqvae.train()
        num_batches = len(data) // batch_size
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_vq = 0.0
            epoch_div = 0.0

            indices = torch.randperm(len(data))
            
            for i in range(num_batches):
                batch = data[indices[i*batch_size : (i+1)*batch_size]]
                
                self.optimizer.zero_grad()
                reconstructed, vq_loss, codes = self.rqvae(batch)
                
                # Debug: Check dimensions
                if i == 0 and epoch == 0:
                    print(f"DEBUG: batch shape: {batch.shape}")
                    print(f"DEBUG: reconstructed shape: {reconstructed.shape}")
                
                # decoder needs diverse codes to reconstruct diverse inputs
                recon_loss = F.mse_loss(reconstructed, batch)
                
                # ADD: Cosine similarity loss (forces semantic preservation)
                cos_sim = F.cosine_similarity(reconstructed, batch, dim=1).mean()
                cos_loss = 1.0 - cos_sim  # Loss when similarity is low
                

                # Add entropy regularization to encourage uniform code usage
                diversity_loss = 0.0
                if diversity_weight > 0:
                    for level in range(codes.shape[1]):
                        level_codes = codes[:, level]
                        # Calculate distribution entropy
                        counts = torch.bincount(level_codes, minlength=self.codebook_sizes[level]).float()
                        probs = counts / counts.sum()
                        # Negative entropy (we want to minimize this, i.e., maximize entropy)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                        # Target is log(codebook_size) for uniform distribution
                        target_entropy = torch.log(torch.tensor(self.codebook_sizes[level], dtype=torch.float32))
                        diversity_loss += (target_entropy - entropy) ** 2
                
                loss = recon_loss + vq_loss + 0.5 * cos_loss + diversity_weight * diversity_loss
                
                loss.backward()

                # ADD GRADIENT CLIPPING (CRITICAL!)
                torch.nn.utils.clip_grad_norm_(self.rqvae.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                
                epoch_recon += recon_loss.item()
                epoch_vq += vq_loss.item()
                epoch_div += diversity_loss.item() if diversity_weight > 0 else 0.0
            
            # Average losses (MOVED OUTSIDE BATCH LOOP)
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            avg_vq = epoch_vq / num_batches
            avg_div = epoch_div / num_batches if diversity_weight > 0 else 0.0
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'recon': f'{avg_recon:.4f}',
                'vq': f'{avg_vq:.4f}',
                'div': f'{avg_div:.4f}' if diversity_weight > 0 else '0'
            })

            # Print every 25 epochs
            if (epoch + 1) % 25 == 0:
                print(f"\nEpoch {epoch+1}:")
                print(f"  Loss: {avg_loss:.6f}")
                print(f"  Recon: {avg_recon:.6f}")
                print(f"  VQ: {avg_vq:.6f}")
                
                # Check unique semantic IDs
                with torch.no_grad():
                    _, _, all_codes = self.rqvae(data)
                    
                    # Convert to tuples
                    semantic_ids = [
                        tuple(c.cpu().numpy().tolist()) 
                        for c in all_codes
                    ]
                    unique_ids = len(set(semantic_ids))
                    
                    print(f"  Unique semantic IDs: {unique_ids}/{len(data)} ({unique_ids/len(data)*100:.1f}%)")
                    mlflow.log_metric("reconstruction_loss", avg_recon, step=epoch)
                    mlflow.log_metric("avg_vq", avg_vq, step=epoch)
                    mlflow.log_metric("avg_recon", avg_recon, step=epoch)
            
                    # Check each level with perplexity
                    for level in range(all_codes.shape[1]):
                        level_codes = all_codes[:, level]
                        unique = len(torch.unique(level_codes))
                        total = self.codebook_sizes[level]
                        
                        # Calculate perplexity
                        counts = torch.bincount(level_codes, minlength=total).float()
                        probs = counts / counts.sum()
                        probs = probs[probs > 0]  # Remove zeros
                        perplexity = torch.exp(-torch.sum(probs * torch.log(probs))).item()
                        
                        print(f"  Level {level+1}: {unique}/{total} ({unique/total*100:.1f}%) | Perplexity: {perplexity:.1f}")

                # Check for actual collapse (very few unique codes, not just low/negative loss)
                # Note: VQ loss can be negative when usage balancing is strong - that's good!
                with torch.no_grad():
                    _, _, all_codes = self.rqvae(data)
                    min_usage = min([
                        len(torch.unique(all_codes[:, level])) / self.codebook_sizes[level]
                        for level in range(all_codes.shape[1])
                    ])
                    
                    if min_usage < 0.05:  # Less than 5% codes used
                        print(f"\n⚠️  Codebook collapsed! (min usage: {min_usage*100:.1f}%)")
                        break


    def inference(self, 
                df, 
                data):
        """
        Generate semantic IDs using the trained model.
        
        Args:
            df: DataFrame with item metadata
            data: The SAME tensor used during training (already normalized)
        """
        print("\nGenerating semantic IDs...")
        self.rqvae.eval()
        
        # Use the data tensor directly (don't re-encode!)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
        else:
            data = data.to(self.device)
        
        with torch.no_grad():
            _, _, codes = self.rqvae(data)
        
        # Immediately check
        print(f"Generated {len(codes)} semantic IDs")
        for level in range(codes.shape[1]):
            unique = len(torch.unique(codes[:, level]))
            print(f"  Level {level+1}: {unique} unique codes")
        
        df = df.copy()
        df['semantic_id'] = [tuple(c.cpu().numpy().tolist()) for c in codes]
        
        # Create string version for sorting and grouping
        df['_semantic_id_str'] = df['semantic_id'].astype(str)
        df = df.sort_values(by=['_semantic_id_str', 'title'])
        df['leaf_id'] = df.groupby('_semantic_id_str').cumcount()
        df = df.drop(columns=['_semantic_id_str'])
        
        df['final_id'] = df.apply(
            lambda x: x['semantic_id'] + (x['leaf_id'],),
            axis=1
        )

        return df