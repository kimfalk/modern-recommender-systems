import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from recsys.semantic_ids.rqvae import RQVAE
from recsys.semantic_ids.vector_quantizer import VectorQuantizerEMA

class SemanticIDPipeline:

    def __init__(self, 
                 codebook_sizes=[8, 32, 128], internal_dim=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading BERT...")
                # CURRENT (fast, lower quality)
        # self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # BETTER (slower, higher quality)
        # self.text_encoder = SentenceTransformer('all-mpnet-base-v2')

        # BEST (slow, best quality)
        self.text_encoder = SentenceTransformer('intfloat/e5-large-v2')
        bert_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        self.codebook_sizes = codebook_sizes
        self.rqvae = RQVAE(bert_dim, internal_dim, codebook_sizes).to(self.device)
        self.optimizer = optim.Adam(self.rqvae.parameters(), lr=1e-3)

    def initialize_data(self, texts):
        print("\nEncoding texts...")
        embeddings = self.text_encoder.encode(texts, 
                                              show_progress_bar=True)
        data = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        # Normalize input embeddings
        #data = F.normalize(data, p=2, dim=1)
        
        print("\nInitializing codebooks...")
        with torch.no_grad():
            latents = self.rqvae.encoder(data)
            residual = latents
            
            for i in range(self.rqvae.depth):
                print(f"\n  Level {i+1}:")
                print(f"    Residual: mean={residual.mean():.4f}, std={residual.std():.4f}")
                
                self.rqvae.quantizers[i].init_codebook(residual)
                z_q, _, _ = self.rqvae.quantizers[i](residual)
                residual = residual - z_q
        
        return data
    
    def train(self,
              data,
              epochs=100, 
              batch_size=64):
            
        # 2. Train Loop
        print("Training RQ-VAE...")
        self.rqvae.train()
        indices = torch.randperm(len(data))
        num_batches = len(data) // batch_size
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_vq = 0.0

            indices = torch.randperm(len(data))
            
            for i in range(num_batches):
                batch = data[indices[i*batch_size : (i+1)*batch_size]]
                
                self.optimizer.zero_grad()
                reconstructed, vq_loss, _ = self.rqvae(batch)
                
                with torch.no_grad():
                    target = self.rqvae.encoder(batch)

                recon_loss = F.mse_loss(reconstructed, target)
                loss = recon_loss + vq_loss
                
                loss.backward()

                # ADD GRADIENT CLIPPING (CRITICAL!)
                torch.nn.utils.clip_grad_norm_(self.rqvae.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                
                epoch_recon += recon_loss.item()
                epoch_vq += vq_loss.item()

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
                    
                    print(f"  Unique semantic IDs: {unique_ids}")
                    
                    # Check each level
                    for level in range(all_codes.shape[1]):
                        level_codes = all_codes[:, level]
                        unique = len(torch.unique(level_codes))
                        total = self.codebook_sizes[level]
                        print(f"  Level {level+1}: {unique}/{total} codes ({unique/total*100:.1f}%)")
                
                # Stop if VQ loss is too low
                if avg_vq < 1e-6:
                    print(f"\n⚠️  VQ loss collapsed! Stopping training.")
                    break
                
            # Average losses
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            avg_vq = epoch_vq / num_batches
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'recon': f'{avg_recon:.4f}',
                'vq': f'{avg_vq:.4f}'
            })

    def inference(self, 
                df, 
                data):
    
        # 3. Inference
        self.rqvae.eval()
        with torch.no_grad():
            _, _, codes = self.rqvae(data)
            
        df['semantic_id'] = [tuple(c) for c in codes.cpu().numpy().tolist()]
        
        # Add Leaf ID
        # Handle both 'title' and 'Title' column names
        title_col = 'Title' if 'Title' in df.columns else 'title'
        # Create temporary string version of semantic_id for sorting and grouping
        df['_semantic_id_str'] = df['semantic_id'].astype(str)
        df = df.sort_values(by=['_semantic_id_str', title_col])
        df['leaf_id'] = df.groupby('_semantic_id_str').cumcount()
        df = df.drop(columns=['_semantic_id_str'])
        df['final_id'] = df.apply(lambda x: x['semantic_id'] + (x['leaf_id'],), axis=1)
        
        return df