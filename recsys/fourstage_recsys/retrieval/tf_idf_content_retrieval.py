
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFContentRetrieval:
    def __init__(self, movies, content_vectors):
        self.movies = movies
        self.content_vectors = content_vectors
        self.movie_id_to_idx = {
            int(mid): idx 
            for idx, mid in enumerate(movies['movieId'])
        } #A

        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()} #A
        self.fit()
    
    def _prepare_data(self):
        self.movies["clean_title"] = (
            self.movies["title"]
            .str.replace(r"\s*\((?:19|20)\d{2}(?:-(?:19|20)\d{2})?\)\s*$", "", regex=True)
            .str.strip()
        )
        self.movies['content'] = self.movies['clean_title'] + ' ' + self.movies['genres'].fillna('')

    def fit(self):
        self._prepare_data()
        
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'
        )

        self.content_vectors = vectorizer.fit_transform(self.movies['content'])


        pass
    def retrieve_similar_by_content(self, movie_id, k=100):
        if movie_id not in self.movie_id_to_idx:
            return []
    
        movie_idx = self.movie_id_to_idx[movie_id] #B
        movie_vector = self.content_vectors[movie_idx] #B
    
        similarities = cosine_similarity(movie_vector, self.content_vectors)[0] #C
    
        top_indices = np.argsort(similarities)[-(k+1):-1][::-1] #D
    
        candidates = []
        for idx in top_indices:
            candidates.append({
                'movie_id': int(self.idx_to_movie_id[idx]),
                'content_similarity': float(similarities[idx])
            })
    
        return candidates