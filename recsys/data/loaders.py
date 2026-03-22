import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

def load_movielens(dataset='ml-100k', data_dir='./data'):
    """
    Download and load MovieLens dataset.
    
    Args:
        dataset: Dataset size ('ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m')
        data_dir: Directory to store the downloaded data
        
    Returns:
        ratings: DataFrame with columns [userId, movieId, rating, timestamp]
        movies: DataFrame with movie information
    """
    base_url = 'https://files.grouplens.org/datasets/movielens/'
    
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_path / dataset
    
    # Download if not exists
    if not dataset_path.exists():
        print(f"Downloading {dataset}...")
        url = f"{base_url}{dataset}.zip"
        response = requests.get(url)
        zip_path = data_path / f"{dataset}.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        os.remove(zip_path)
        print(f"Downloaded and extracted {dataset}")
    
    # Load ratings
    if dataset == 'ml-100k':
        ratings = pd.read_csv(
            dataset_path / 'u.data',
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp']
        )
        movies = pd.read_csv(
            dataset_path / 'u.item',
            sep='|',
            encoding='latin-1',
            names=['movieId', 'title', 'release_date', 'video_release_date',
                   'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )
    else:
        # ml-1m, ml-10m, ml-20m, ml-25m format
        ratings = pd.read_csv(
            dataset_path / 'ratings.csv'
        )
        movies = pd.read_csv(
            dataset_path / 'movies.csv'
        )
    
    return ratings, movies


def load_movielens_descriptions(data_dir='./data', auto_download=True):
    """
    Load MovieLens movie descriptions/plot summaries from HuggingFace.
    
    Downloads movie plot summaries from HuggingFace datasets if not already present.
    
    Args:
        data_dir: Directory where description data is stored
        auto_download: If True, automatically download descriptions if not found
        
    Returns:
        DataFrame with movie information including descriptions
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    descriptions_path = data_path / 'movie_descriptions.csv'
    
    # Download if not exists and auto_download is enabled
    if not descriptions_path.exists() and auto_download:
        print("Downloading movie descriptions from HuggingFace...")
        try:
            from datasets import load_dataset
            
            # Load dataset from HuggingFace
            dataset = load_dataset("Pablinho/movies-dataset")
            
            # Convert to pandas DataFrame
            df = dataset['train'].to_pandas()
            
            # Save to CSV for future use
            df.to_csv(descriptions_path, index=False)
            print(f"Downloaded and saved movie descriptions to {descriptions_path}")
            print(f"Loaded {len(df)} movie descriptions")
            print(f"{df['title'].isna().sum()} descriptions missing titles")
            return df
            
        except ImportError:
            print("HuggingFace datasets library not installed. Install with: pip install datasets")
            return pd.DataFrame()
        except Exception as e:
            print(f"Failed to download descriptions: {e}")
            print("Returning empty DataFrame...")
            return pd.DataFrame()
    
    if descriptions_path.exists():
        try:
            descriptions = pd.read_csv(descriptions_path)
            print(f"Loaded {len(descriptions)} movie descriptions from cache")
            return descriptions
            
        except Exception as e:
            print(f"Error loading descriptions: {e}")
            return pd.DataFrame()
    else:
        print(f"No descriptions file found at {descriptions_path}")
        return pd.DataFrame()

