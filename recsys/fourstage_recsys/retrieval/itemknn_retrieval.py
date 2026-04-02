import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class ItemKNNRetrieval:
    def __init__(self, ratings):
        self.ratings = ratings
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.user_item_matrix = None
        self.item_similarity = None
        
        self._prepare_data()
        self._compute_item_similarity()
    
    def _compute_item_similarity(self):
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        print(f"item_similarity shape: {self.item_similarity.shape}")
        
    def _prepare_data(self):
        user_ids = self.ratings['userId'].unique()
        movie_ids = self.ratings['movieId'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)} #A
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)} #A
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()} #A

        self.rows = [self.user_to_idx[uid] for uid in self.ratings['userId']] #B
        self.cols = [self.movie_to_idx[mid] for mid in self.ratings['movieId']] #B
        self.data = [1] * len(self.ratings) #B

        self.user_item_matrix = csr_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(user_ids), len(movie_ids))
        ) #C
        print(f"user_item_matrix shape: {self.user_item_matrix.shape}")
    
    def retrieve_similar_items(self, movie_id, k=100) -> list[dict]: #A
        if movie_id not in self.movie_to_idx:
            return []
    
        movie_idx = self.movie_to_idx[movie_id]
        similarities = self.item_similarity[movie_idx]
        
        top_indices = np.argsort(similarities)[-(k+1):-1][::-1]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'movie_id': self.idx_to_movie[idx],
                'similarity': float(similarities[idx])
            })
        
        return candidates