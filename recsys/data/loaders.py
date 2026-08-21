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
    ratings['userId'] = ratings['userId'].astype(str)
    ratings['movieId'] = ratings['movieId'].astype(str)
    movies['movieId'] = movies['movieId'].astype(str)
        
    return ratings, movies

def load_movielens_links(dataset='ml-100k', data_dir='./data'):
    """
    Load MovieLens links data which contains TMDB IDs for movies.
    
    Args:
        dataset: Dataset size ('ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m')
        data_dir: Directory where the data is stored
    Returns:
        links: DataFrame with columns [movieId, imdbId, tmdbId]
    """
    data_path = Path(data_dir) / dataset
    links_path = data_path / 'links.csv'
    
    if links_path.exists():
        try:
            links = pd.read_csv(links_path)
            print(f"Loaded links data from {links_path}")
            return links
        except Exception as e:
            print(f"Error loading links data: {e}")
            return pd.DataFrame()
    else:
        print(f"No links file found at {links_path}")
        return pd.DataFrame()
    

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

def load_movielens_ratings(data_dir='./data', auto_download=True):
    """
    Load MovieLens movie ratings from CSV files.
    
    Downloads movie ratings from HuggingFace datasets if not already present.
    
    Args:
        data_dir: Directory where rating data is stored
        auto_download: If True, automatically download ratings if not found
        
    Returns:
        DataFrame with movie ratings
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    ratings_path = data_path / 'ratings.csv'
    
    # Download if not exists and auto_download is enabled
    if not ratings_path.exists() and auto_download:
        print("Downloading movie ratings from HuggingFace...")
        try:
            from datasets import load_dataset
            
            # Load dataset from HuggingFace
            dataset = load_dataset("Pablinho/movies-dataset")
            
            # Convert to pandas DataFrame
            df = dataset['train'].to_pandas()
            
            # Save to CSV for future use
            df.to_csv(ratings_path, index=False)
            print(f"Downloaded and saved movie ratings to {ratings_path}")
            print(f"Loaded {len(df)} movie ratings")
            return df
            
        except ImportError:
            print("HuggingFace datasets library not installed. Install with: pip install datasets")
            return pd.DataFrame()
        except Exception as e:
            print(f"Failed to download ratings: {e}")
            print("Returning empty DataFrame...")
            return pd.DataFrame()
    
    if ratings_path.exists():
        try:
            ratings = pd.read_csv(ratings_path)
            print(f"Loaded {len(ratings)} movie ratings from cache")
            return ratings
            
        except Exception as e:
            print(f"Error loading ratings: {e}")
            return pd.DataFrame()
    else:
        print(f"No ratings file found at {ratings_path}")
        return pd.DataFrame()


def load_tmdb_movie_descriptions(
    links,
    api_key=None,
    data_dir='./data',
    cache_filename='movielens_descriptions.csv',
    force_refresh=False,
    request_delay=0.05,
):
    """
    Load movie descriptions from TMDB, using a CSV cache when available.

    Args:
        links: DataFrame with at least columns [movieId, tmdbId] (e.g. from
            ``load_movielens_links``).
        api_key: TMDB API key. Required only when fetching new descriptions.
        data_dir: Directory where the cache CSV lives.
        cache_filename: Name of the cache CSV inside ``data_dir``.
        force_refresh: If True, ignore the cache and re-fetch from TMDB.
        request_delay: Seconds to sleep between TMDB requests.

    Returns:
        dict mapping ``movieId`` (int) to a dict with keys
        ``{'title', 'overview', 'genres'}``.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    cache_path = data_path / cache_filename

    descriptions = {}

    if cache_path.exists() and not force_refresh:
        try:
            cached_df = pd.read_csv(cache_path)
            for _, row in cached_df.iterrows():
                movie_id = int(row['movieId'])
                descriptions[movie_id] = {
                    'title': row['title'] if pd.notna(row.get('title')) else '',
                    'overview': row['overview'] if pd.notna(row.get('overview')) else '',
                    'genres': row['genres'] if pd.notna(row.get('genres')) else '',
                }
            print(f"Loaded {len(descriptions)} cached descriptions from {cache_path}")
        except Exception as e:
            print(f"Error loading description cache: {e}")
            descriptions = {}

    if links is None or len(links) == 0:
        return descriptions

    wanted_ids = set(int(m) for m in links['movieId'].dropna().tolist())
    missing_ids = wanted_ids - set(descriptions.keys())

    if not missing_ids:
        return descriptions

    if not api_key:
        print(
            f"{len(missing_ids)} descriptions missing but no TMDB API key "
            "provided; returning cached descriptions only."
        )
        return descriptions

    import time

    tmdb_lookup = (
        links.dropna(subset=['tmdbId'])
        .assign(movieId=lambda df: df['movieId'].astype(int))
        .set_index('movieId')['tmdbId']
        .to_dict()
    )

    base_url = 'https://api.themoviedb.org/3/movie/{tmdb_id}'
    print(f"Fetching {len(missing_ids)} descriptions from TMDB...")

    fetched = 0
    for i, movie_id in enumerate(sorted(missing_ids), start=1):
        tmdb_id = int(tmdb_lookup.get(movie_id))
        if tmdb_id is None or pd.isna(tmdb_id):
            continue
        try:
            resp = requests.get(
                base_url.format(tmdb_id=int(tmdb_id)),
                params={'api_key': api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                descriptions[movie_id] = {
                    'title': data.get('title', '') or '',
                    'overview': data.get('overview', '') or '',
                    'genres': '|'.join(
                        g.get('name', '') for g in data.get('genres', [])
                    ),
                }
                fetched += 1
            elif resp.status_code == 404:
                continue
            else:
                print(
                    f"TMDB request failed (status={resp.status_code}) "
                    f"for movieId={movie_id}, tmdbId={tmdb_id}"
                )
            time.sleep(request_delay)
        except Exception as e:
            print(f"Error fetching tmdbId={tmdb_id}: {e}")
            continue

        if fetched and fetched % 500 == 0:
            _save_tmdb_descriptions_cache(descriptions, cache_path)
            print(f"  ... saved intermediate cache ({len(descriptions)} entries)")

    _save_tmdb_descriptions_cache(descriptions, cache_path)
    print(f"Saved {len(descriptions)} descriptions to {cache_path}")
    return descriptions


def _save_tmdb_descriptions_cache(descriptions, cache_path):
    """Persist a movieId -> description dict to CSV."""
    if not descriptions:
        return
    df = pd.DataFrame.from_dict(descriptions, orient='index')
    df.reset_index(names='movieId', inplace=True)
    df.to_csv(cache_path, index=False)
