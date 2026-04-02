
from abc import ABC, abstractmethod
from typing import List, Any

from pathlib import Path
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem

class Ordering(ABC):
  @abstractmethod
  def rank(
    self, 
    filtered_items: List[ScoredItem],
    context: RecommendationContext
  ) -> List[ScoredItem]:
    pass
