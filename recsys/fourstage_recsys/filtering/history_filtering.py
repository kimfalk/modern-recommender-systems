"""
recsys/fourstage_recsys/filtering/history_filtering.py
======================================================
Filtering stage that removes items the user has already interacted with and
deduplicates the candidate list.

Accepts either a ratings DataFrame (as in Chapter 2) or a precomputed
per-user item dict (as produced by ``recsys.data.preprocessing
.user_item_lists``). Histories are indexed once at construction time, so
filtering is a set lookup per request instead of a DataFrame scan.

Conforms to the ``Filtering`` ABC: ``filter(scored_items, context)`` receives
the ``RecommendationContext``, which is what ``FourStageRecommender`` passes.
For backward compatibility, a bare user ID in place of the context also
works.
"""

from typing import List

import pandas as pd

from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.stages.filtering import Filtering


class HistoryFiltering(Filtering):
    """Remove already-seen items and duplicates from the candidate list."""

    def __init__(self, history):
        if isinstance(history, pd.DataFrame):                   #A
            ordered = history.sort_values(["userId", "timestamp"])
            self._history = (ordered.groupby("userId")["movieId"]
                             .apply(list).to_dict())
        else:                                                   #B
            self._history = {user: list(items)
                             for user, items in history.items()}
        self._seen = {user: set(items)
                      for user, items in self._history.items()}

    def get_user_history(self, user_id, k=None) -> list:
        items = self._history.get(user_id, [])
        return items if k is None else items[-k:]               #C

    def filter(self, scored_items: List[ScoredItem],
               context) -> List[ScoredItem]:
        user_id = getattr(context, "user_id", context)          #D
        seen = self._seen.get(user_id, set())
        out, taken = [], set()
        for item in scored_items:
            if item.item_id in seen or item.item_id in taken:   #E
                continue
            out.append(item)
            taken.add(item.item_id)
        return out

#A A ratings DataFrame is indexed into per-user chronological histories once
#B A dict of user -> items (e.g. from user_item_lists) is used as-is
#C The k most recent items, preserving chronological order
#D Works with a RecommendationContext or a bare user ID
#E Drop already-seen items and deduplicate merged retrieval sources