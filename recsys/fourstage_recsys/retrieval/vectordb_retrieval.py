from typing import List, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext

class VectorRetrieval(Retrieval):
  
  def __init__(self, index_path: Path, embeddings_path: Path):
    import faiss
    self.index = faiss.read_index(str(index_path))
    self.embeddings = np.load(embeddings_path)
    self.item_ids = np.load(embeddings_path.parent / "item_ids.npy")
  
  def retrieve(self, context: RecommendationContext) -> List[ScoredItem]:
    query_vector = self._get_query_vector(context)
    
    if query_vector is None:
      return []
    
    distances, indices = self.index.search(
      query_vector.reshape(1, -1), 
      k=100
    )
    
    return [ScoredItem(item_id=str(self.item_ids[idx]), score={"distance": distances[0][i]}) for i, idx in enumerate(indices[0])]
  
  def _get_query_vector(
    self, 
    context: RecommendationContext
  ) -> Optional[np.ndarray]:
    if context.item_id:
      item_idx = np.where(self.item_ids == context.item_id)[0]
      if len(item_idx) > 0:
        return self.embeddings[item_idx[0]]
    
    return None
#A Load FAISS index and embeddings #
#B Get query vector based on context 
#C Perform approximate nearest neighbor search 
#D Return top 100 similar items

