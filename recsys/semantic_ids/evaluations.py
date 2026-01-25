import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np

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

def test_semantic_coherence(df):
  """
  Verifies that known similar items have similar semantic IDs.
  """
  test_pairs = [ #A Define test pairs: (item1, item2, expected_similarity)
    ("Star Wars", "Star Trek", "high"),#B Should be similar (same genre)
    ("The Notebook", "Pride and Prejudice", "high"),#B Should be similar (same genre)
    ("Die Hard", "Lethal Weapon", "high"),#B Should be similar (same genre)        
    ("Star Wars", "The Notebook", "low"), #C Should be dissimilar (different genres)
    ("Die Hard", "Frozen", "low"),#C Should be dissimilar (different genres)
    ("Inception", "The Lion King", "low"),#C Should be dissimilar (different genres)
  ] 

  results = []
  for title1, title2, expected in test_pairs:
    # Find semantic IDs
    try:
      id1 = df[df['title'] == title1]['semantic_id'].values[0]
      id2 = df[df['title'] == title2]['semantic_id'].values[0]
    except IndexError:
      print(f"Warning: Could not find '{title1}' or '{title2}'")
      continue
        
    # Count matching levels
    matches = sum(a == b for a, b in zip(id1, id2)) #A
        
    # Calculate similarity score (0 to 1)
    similarity = matches / len(id1) #B
        
    # Check if result matches expectation
    if expected == "high":
      passed = similarity >= 0.5  # At least 50% match
    else:
      passed = similarity < 0.5
        
    status = "✓" if passed else "✗"
    results.append({
      'pair': f"{title1} vs {title2}",
      'matches': matches,
      'similarity': similarity,
      'expected': expected,
      'passed': passed
    })    
    print(f"{status} {title1:30s} vs {title2:30s}: "
    f"{matches}/{len(id1)} levels match (expected: {expected})")
    
    # Summary
    passed_count = sum(r['passed'] for r in results)
    print(f"\nPassed: {passed_count}/{len(results)} tests")
    
    return results

#A Count how many levels of the semantic ID match
#B Convert to 0-1 similarity score

def analyze_codebook_usage(df, codebook_sizes):
    """
    Checks if all codes in each codebook are being used.
    """
    import matplotlib.pyplot as plt
    
    for level in range(len(codebook_sizes)):
        # Extract codes at this level
        codes = df['semantic_id'].apply(lambda x: x[level]).values
        
        # Count usage
        unique_codes, counts = np.unique(codes, return_counts=True)
        
        usage_pct = len(unique_codes) / codebook_sizes[level] * 100
        
        print(f"\nLevel {level + 1} (Codebook size: {codebook_sizes[level]}):")
        print(f"  Used codes: {len(unique_codes)}/{codebook_sizes[level]} "
              f"({usage_pct:.1f}%)")
        print(f"  Min usage: {counts.min()} items")
        print(f"  Max usage: {counts.max()} items")
        print(f"  Mean usage: {counts.mean():.1f} items")
        
        # Visualize distribution
        plt.figure(figsize=(10, 4))
        plt.bar(unique_codes, counts)
        plt.xlabel(f'Code Index (Level {level + 1})')
        plt.ylabel('Number of Items')
        plt.title(f'Code Usage Distribution - Level {level + 1}')
        plt.tight_layout()
        plt.savefig(f'codebook_usage_level_{level + 1}.png')
        plt.close()
        
        # Warning for dead codes
        if usage_pct < 80:
            print(f"  ⚠️  Warning: Only {usage_pct:.1f}% of codes used. "
                  f"Consider reducing codebook size.")

def evaluate_clustering(df, embeddings):
    """
    Measures how well semantic IDs cluster similar items.
    """
    # Extract top-level codes (first element of semantic ID)
    top_level_codes = df['semantic_id'].apply(lambda x: x[0]).values #A
    
    # Silhouette Score: measures cluster cohesion and separation
    silhouette = silhouette_score(embeddings, top_level_codes) #B
    
    # Calinski-Harabasz: ratio of between-cluster to within-cluster variance
    calinski = calinski_harabasz_score(embeddings, top_level_codes) #C
    
    print(f"Clustering Metrics (Top-Level Codes):")
    print(f"  Silhouette Score: {silhouette:.3f} (range: -1 to 1, target > 0.3)")
    print(f"  Calinski-Harabasz: {calinski:.1f} (higher is better)")
    
    # Also check second level
    second_level = df['semantic_id'].apply(
        lambda x: f"{x[0]}_{x[1]}"
    ).values
    silhouette_l2 = silhouette_score(embeddings, second_level)
    
    print(f"\nClustering Metrics (Second-Level Codes):")
    print(f"  Silhouette Score: {silhouette_l2:.3f}")
    
    return {
        'silhouette_l1': silhouette,
        'silhouette_l2': silhouette_l2,
        'calinski_harabasz': calinski
    }

#A Use only the first code for top-level clustering analysis
#B Silhouette: -1 (wrong clusters) to 1 (perfect clusters)
#C Calinski-Harabasz: higher values indicate better-defined clusters

def evaluate_semantic_ids(pipeline, df, embeddings, codebook_sizes):
    """
    Runs all evaluation metrics and generates a report.
    """
    print("="*60)
    print("SEMANTIC ID EVALUATION REPORT")
    print("="*60)
    
    # 1. Reconstruction Quality
    print("\n1. RECONSTRUCTION QUALITY")
    print("-"*60)
    recon_metrics = evaluate_reconstruction(pipeline, embeddings)
    
    # 2. Clustering Quality
    print("\n2. CLUSTERING QUALITY")
    print("-"*60)
    cluster_metrics = evaluate_clustering(df, embeddings.cpu().numpy())
    
    # 3. Semantic Coherence
    print("\n3. SEMANTIC COHERENCE")
    print("-"*60)
    coherence_results = test_semantic_coherence(df)
    
    # 4. Codebook Utilization
    print("\n4. CODEBOOK UTILIZATION")
    print("-"*60)
    analyze_codebook_usage(df, codebook_sizes)
    
    # Overall Assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    
    # Define passing criteria
    passes = []
    
    # Reconstruction
    if recon_metrics['cosine_similarity'] > 0.9:
        print("✓ Reconstruction: EXCELLENT")
        passes.append(True)
    elif recon_metrics['cosine_similarity'] > 0.8:
        print("⚠ Reconstruction: GOOD")
        passes.append(True)
    else:
        print("✗ Reconstruction: POOR - Consider larger codebooks")
        passes.append(False)
    
    # Clustering
    if cluster_metrics['silhouette_l1'] > 0.4:
        print("✓ Clustering: EXCELLENT")
        passes.append(True)
    elif cluster_metrics['silhouette_l1'] > 0.3:
        print("⚠ Clustering: ACCEPTABLE")
        passes.append(True)
    else:
        print("✗ Clustering: POOR - Check encoder quality")
        passes.append(False)
    
    # Coherence
    coherence_pass_rate = sum(r['passed'] for r in coherence_results) / len(coherence_results)
    if coherence_pass_rate > 0.8:
        print("✓ Semantic Coherence: EXCELLENT")
        passes.append(True)
    elif coherence_pass_rate > 0.6:
        print("⚠ Semantic Coherence: ACCEPTABLE")
        passes.append(True)
    else:
        print("✗ Semantic Coherence: POOR - Review text encoding strategy")
        passes.append(False)
    
    if all(passes):
        print("\n🎉 Semantic IDs are ready for production!")
    elif sum(passes) >= 2:
        print("\n⚠️  Semantic IDs are acceptable but could be improved.")
    else:
        print("\n❌ Semantic IDs need significant improvement before production.")
    
    return {
        'reconstruction': recon_metrics,
        'clustering': cluster_metrics,
        'coherence': coherence_results,
        'overall_pass': all(passes)
    }
