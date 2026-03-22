from abc import ABC, abstractmethod
from typing import List, Any

class Filtering(ABC):
  
  @abstractmethod
  def filter(
    self, 
    scored_items: List[ScoredItem],
    context: RecommendationContext
  ) -> List[ScoredItem]:
    pass
