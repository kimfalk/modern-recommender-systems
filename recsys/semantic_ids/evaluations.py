import torch
import torch.nn.functional as F

def evaluate_reconstruction(pipeline, data_tensor):
    """
    Measures how well the RQ-VAE can reconstruct latent representations
    from semantic IDs.
    """
    pipeline.rqvae.eval()
    with torch.no_grad():
        # Get the latent representation (what the encoder produces)
        latents = pipeline.rqvae.encoder(data_tensor) #A
        
        # Pass through quantizers and decoder
        reconstructed, vq_loss, codes = pipeline.rqvae(data_tensor) #B
        
        # Compare reconstructed latents to original latents
        mse = F.mse_loss(reconstructed, latents) #C
        
        # Cosine Similarity (more interpretable)
        cos_sim = F.cosine_similarity(
            reconstructed, 
            latents
        ).mean() #D
        
        # Per-dimension correlation
        correlation = torch.corrcoef(
            torch.stack([reconstructed.flatten(), 
                        latents.flatten()])
        )[0, 1] #E
        
    print(f"Reconstruction Metrics (Latent Space):")
    print(f"  MSE: {mse:.4f} (lower is better, target < 0.1)")
    print(f"  Cosine Similarity: {cos_sim:.4f} (higher is better, target > 0.9)")
    print(f"  Correlation: {correlation:.4f} (target > 0.85)")
    
    return {
        'mse': mse.item(),
        'cosine_similarity': cos_sim.item(),
        'correlation': correlation.item()
    }