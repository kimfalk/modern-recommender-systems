import pandas as pd
from pathlib import Path
import numpy as np

from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext

from sqlalchemy import create_engine
from typing import List, Optional

class DatabaseRetrieval(Retrieval):
  
  def __init__(self, connection_string: str):
    self.engine = create_engine(connection_string)
  
  def retrieve(self, context: RecommendationContext) -> List[ScoredItem]:
    if context.item_id:
      return self._retrieve_similar_items(context.item_id)
    elif context.user_id:
      return self._retrieve_for_user(context.user_id)
    else:
      return self._retrieve_popular()
  
  def _retrieve_similar_items(self, item_id: str) -> List[ScoredItem]:
    query = """
      SELECT similar_item_id, similarity
      FROM item_similarities 
      WHERE item_id = %s 
      ORDER BY similarity DESC 
      LIMIT 100
    """
    with self.engine.connect() as conn:
      result = conn.execute(query, (item_id,))
      return [ScoredItem(item_id=row[0], score={"similarity": row[1]}) for row in result]
  
  def _retrieve_for_user(self, user_id: str) -> List[ScoredItem]:
    query = """
      SELECT item_id, score
      FROM user_recommendations 
      WHERE user_id = %s 
      ORDER BY score DESC 
      LIMIT 100
    """
    with self.engine.connect() as conn:
      result = conn.execute(query, (user_id,))
      return [ScoredItem(item_id=row[0], score={"score": row[1]}) for row in result]
  
  def _retrieve_popular(self) -> List[ScoredItem]:
    query = """
      SELECT item_id, popularity_score
      FROM popular_items 
      ORDER BY popularity_score DESC 
      LIMIT 100
    """
    with self.engine.connect() as conn:
      result = conn.execute(query)
      return [ScoredItem(item_id=row[0], score={"popularity_score": row[1]}) for row in result]
#A Connect to database #B Route to appropriate query based on context #C Query for items similar to seed item #D Query for user's personalized candidates #E Query for popular items (fallback)
