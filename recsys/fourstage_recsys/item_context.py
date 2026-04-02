from abc import ABC, abstractmethod
from typing import List, Any, Dict
from pathlib import Path
from pydantic import BaseModel

class ScoredItem(BaseModel):
  item_id: str #A
  score: Dict[str, float] = {} #B
  metadata: Dict[str, Any] = {} #C
  class Config:
    arbitrary_types_allowed = True
