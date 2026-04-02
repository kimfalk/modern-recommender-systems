
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.stages.filtering import Filtering
from recsys.fourstage_recsys.stages.scoring import Scoring

class FourStageRecommender:
    def __init__(
        self,
        retrieval,
        filter: Filtering,          
        scorer: Scoring,
        ranker,
    ):
        self.retrieval = retrieval
        self.filter = filter
        self.scorer = scorer
        self.ranker = ranker

    def recommend(self, context: RecommendationContext):
        
        candidates = self.retrieval.retrieve_similar_items(context.seed_movie_id, k=100)
        candidates = self.filter.filter(candidates, context.user_id)
        candidates = self.scorer.score(candidates)
        ranked = self.ranker.rank(candidates)
        
        return ranked[:context.k]
