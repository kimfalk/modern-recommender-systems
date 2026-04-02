from typing import Optional, Dict, Any
from pydantic import BaseModel
class RecommendationContext(BaseModel):
  user_id: Optional[str] = None #A
  item_id: Optional[str] = None #B
  k: int = 10 #C
  filters: Dict[str, Any] = {}
  metadata: Dict[str, Any] = {}
  class Config:
    arbitrary_types_allowed = True
