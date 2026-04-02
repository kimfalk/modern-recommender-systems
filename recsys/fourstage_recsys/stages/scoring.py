from abc import ABC, abstractmethod
from typing import List, Any

class Scorer(ABC):
  @abstractmethod
  def score(
    self, 
    candidates: List[ScoredItem], 
    context: RecommendationContext
  ) -> List[ScoredItem]:
    pass
