import pandas as pd
from pathlib import Path
import numpy as np
from typing import List

from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext

class FileBasedRetrieval(Retrieval):
  
  def __init__(self, 
               interactions_path: Path, 
               items_path: Path,
               context: RecommendationContext):
    self.context = context
    self.interactions_df = pd.read_csv(interactions_path)
    self.items_df = pd.read_csv(items_path)
    self._prepare()
  
  def _prepare(self):
    pass
  
  def retrieve(self, context: RecommendationContext) -> List[ScoredItem]:
    pass
#A Load data from CSV files 
#B Prepare any needed data structures 
#C Implement retrieval logic (to be filled in by subclasses)
