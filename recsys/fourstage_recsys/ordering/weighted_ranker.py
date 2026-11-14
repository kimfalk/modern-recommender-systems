from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.stages.ordering import Ordering


class WeightedRanker(Ordering):
  def __init__(self, weights=None):
    """
    Rank candidates by a weighted combination of scores.

    Args:
        weights: dict mapping score names to weights, e.g.
                 {'similarity': 0.7, 'popularity': 0.3}
    """
    self.weights = weights or {"similarity": 0.7, "popularity": 0.3}

  def order(
    self, candidates: list[ScoredItem], context: RecommendationContext, debug=False
  ) -> list[ScoredItem]:

    for item in candidates:
      item.scores["final_score"] = sum(
        weight * item.scores.get(score_name, 0.0)
        for score_name, weight in self.weights.items()
      )  # A

    ranked = sorted(
      candidates, key=lambda x: x.scores["final_score"], reverse=True
    )  # B

    if debug:
      print("Candidates with final scores:")
      for item in ranked[:5]:  # Print top 5 for brevity
        print(
          f"Item {item.item_id}: final_score={item.scores['final_score']:.3f}, scores={item.scores}"
        )

    return ranked[: context.k] if context.k else ranked


# A Compute weighted combination of all available scores
# B Sort by final score descending
