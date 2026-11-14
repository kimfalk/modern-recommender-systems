"""
recsys/fourstage_recsys/scoring/passthrough.py
==============================================
A scoring stage that scores nothing.

Retrieval order is the null hypothesis for the scoring stage. Running it
through the same ``FourStageRecommender`` as the real scorers -- same
retrieval, same filtering, ordering by the retrieval ``similarity`` score --
keeps the comparison on identical code paths.
"""

from __future__ import annotations

from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.stages.scoring import Scorer


class PassThroughScoring(Scorer):
  """Leave the candidates and their retrieval scores untouched."""

  def score(
    self, candidates: list[ScoredItem], context: RecommendationContext
  ) -> list[ScoredItem]:
    return candidates
