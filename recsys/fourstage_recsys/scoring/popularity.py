class PopularityScoring:
    def __init__(self, ratings):
        movie_counts = ratings['movieId'].value_counts() #A
        max_count = movie_counts.max() #B

        self.popularity_scores = {
            int(mid): float(count / max_count)
            for mid, count in movie_counts.items()
        }
    
    def score(self, movie_id):
        return self.popularity_scores.get(movie_id, 0.0)
    
    def score_popularity(self, candidates):
        for item in candidates:
            item['popularity'] = self.score(item['movie_id'])
        return candidates