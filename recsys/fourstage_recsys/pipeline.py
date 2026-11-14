from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.stages.filtering import Filtering
from recsys.fourstage_recsys.stages.scoring import Scorer
from recsys.fourstage_recsys.stages.ordering import Ordering

class FourStageRecommender:
    def __init__(
        self,
        retrieval: Retrieval,
        filter: Filtering,          
        scorer: Scorer,
        ordering: Ordering,
        retrieval_k: int = 100,
    ):
        self.retrieval = retrieval
        self.filter = filter
        self.scorer = scorer
        self.ordering = ordering
        self.retrieval_k = retrieval_k
 
    def recommend(self, context: RecommendationContext, debug=False):
        """Run the four-stage pipeline: 
            retrieve → filter → score → order.
        """
        
        candidates = self.retrieval.retrieve_similar_items(
            context.seed_items, k=self.retrieval_k) #A
        if debug:
            print(f"Retrieved {len(candidates)} candidates")
        candidates = self.filter.filter(candidates, context) #B
        if debug:
            print(f"Filtered down to {len(candidates)} candidates")
        candidates = self.scorer.score(candidates, context) #C
        if debug:
            print(f"Scored {len(candidates)} candidates")
        ordered = self.ordering.order(candidates, context, debug=debug) #D
        if debug:
            print(f"Ordered {len(ordered)} candidates")
        
        return ordered[:context.k]
 
#A Stage 1: Retrieval
#B Stage 2: Filtering
#C Stage 3: Scoring
#D Stage 4: Ordering