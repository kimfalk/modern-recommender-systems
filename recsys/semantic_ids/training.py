import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from recsys.semantic_ids.semantic_ids_pipeline import SemanticIDPipeline
from recsys.semantic_ids.utils import prepare_data
from recsys.semantic_ids.evaluations import evaluate_reconstruction, evaluate_semantic_ids

def run_pipeline_debug(df,
                    codebook_sizes = [8, 16, 64],  # Smaller for debugging
                    internal_dim = 256,
                    epochs = 200):
    """Minimal version for debugging."""
    
    print(f"Processing {len(df)} items\n")
    
    # SIMPLE configuration
    codebook_sizes = [8, 16, 64]  # Smaller for debugging
    internal_dim = 256  # Smaller
    epochs = 200
    
    # Create pipeline
    pipeline = SemanticIDPipeline(
        codebook_sizes=codebook_sizes,
        internal_dim=internal_dim
    )
    
    # Prepare data
    texts = prepare_data(df)
    
    # Initialize
    print("Initializing...")
    data_tensor = pipeline.initialize_data(texts)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    pipeline.train(data_tensor, epochs=epochs, batch_size=64)
    
    # Generate IDs
    print("\nGenerating semantic IDs...")
    df_enriched = pipeline.inference(df, data_tensor)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Check codebook usage
    for level in range(len(codebook_sizes)):
        level_codes = df_enriched['semantic_id'].apply(lambda x: x[level]).values
        unique = len(np.unique(level_codes))
        total = codebook_sizes[level]
        print(f"Level {level+1}: {unique}/{total} codes used ({unique/total*100:.1f}%)")
    
    return df_enriched, pipeline, data_tensor

if __name__ == "__main__":
    print("Let's get this party started!\n",
          "Running Semantic ID Pipeline Debug Version\n")
    simple_data = {
        'item_id': [1, 2, 3, 4, 5],
        'title': [
            "The Great Adventure",
            "Mystery of the Old House",
            "Secrets of the Forest",
            "Legends of the Sea"
        ],
        'description': [
            "An epic tale of adventure and discovery.",
            "A thrilling mystery set in an old mansion.",
            "Exploring uncharted territories and facing the unknown.",
            "Unveiling the secrets hidden within the forest.",
            "Stories of bravery and legend from the high seas."
        ],
        'genres': [
            "Adventure, Action",
            "Mystery, Thriller",
            "Adventure, Sci-Fi",
            "Fantasy, Mystery",
            "Adventure, Drama"
        ],
        'format': [
            "Book",
            "Book",
            "Book",
            "Book",
            "Book"
        ]
    }
    # Load simple dataset
    simple_titles_df = pd.DataFrame(simple_data)
    # Run debug version
    df_enriched, pipeline = run_pipeline_debug(simple_titles_df)