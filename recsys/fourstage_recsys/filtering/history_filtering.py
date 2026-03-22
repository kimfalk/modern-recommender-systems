import pandas as pd
import numpy as np

from recsys.fourstage_recsys.stages.filtering import Filtering

class HistoryFiltering(Filtering): #A
    def __init__(self, ratings):
        self.ratings = ratings

    def get_user_history(self, user_id, k=None) -> int: #A
        user_rows = self.ratings[self.ratings["userId"] == user_id]
        user_rows = user_rows.sort_values("timestamp")
        movie_ids = user_rows["movieId"].tolist()
        return set(movie_ids) if k is None else set(movie_ids[-k:])

    def filter(self, candidates, user_id) -> list[dict]: #B
        user_history = self.get_user_history(user_id)
        
        filtered = [
            item for item in candidates
            if item['movie_id'] not in user_history
        ]    
        return filtered