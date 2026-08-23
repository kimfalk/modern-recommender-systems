"""
recsys/fourstage_recsys/retrieval/ann_retrieval.py
==================================================
Approximate Nearest Neighbor retrieval with FAISS (Chapter 5, Section 5.2).

Contains the minimal build/query functions (Listing 5.4), the
production-shaped ``ANNRetrievalIndex`` (index configuration, incremental
additions, recall measurement against exact search, switching among flat,
HNSW, and IVF), the ``ANNRetrieval`` stage that plugs into the four-stage
pipeline, and the ``warm_start_embedding`` cold-start strategy
(Section 5.2.4).

Two invariants this module is pedantic about:

  - **Normalize consistently** between training, index construction, and
    query time. With L2-normalized vectors the inner product equals cosine
    similarity; failing to normalize at any one stage degrades quality
    silently.
  - **Set the FAISS metric explicitly.** ``faiss.IndexHNSWFlat(dim, m)``
    defaults to ``METRIC_L2``. With normalized vectors the *ranking* is
    identical (squared L2 distance is 2 - 2*cos), so the bug is invisible in
    recall metrics -- but the returned scores are distances (smaller is
    better), which silently breaks anything downstream that treats them as
    similarities.

Usage
-----
from recsys.fourstage_recsys.retrieval.ann_retrieval import (
    build_index, query_index, ANNRetrievalIndex, ANNRetrieval,
    warm_start_embedding,
)
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

# On macOS, pip-installed torch and faiss each bundle their own OpenMP
# runtime (libomp.dylib); loading both aborts the process ("OMP: Error #15"),
# which Jupyter reports only as a kernel crash. Allowing the duplicate and
# keeping FAISS single-threaded avoids both the abort and the segfault that
# follows when the two runtimes share threads.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)

from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.retrieval.retrieval import Retrieval


# ---------------------------------------------------------------------------
# Minimal build and query (Listing 5.4)
# ---------------------------------------------------------------------------

def build_index(item_embeddings: np.ndarray) -> faiss.Index:
    """Build an HNSW index over normalized item embeddings."""
    embeddings = np.ascontiguousarray(item_embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)                          #A
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)  #B
    index.hnsw.efConstruction = 200                         #C
    index.hnsw.efSearch = 500                               #D
    index.add(embeddings)                                   #E
    return index


def query_index(index: faiss.Index, query_embedding: np.ndarray,
                k: int = 500):
    """Query the index with one embedding; returns (ids, scores)."""
    query = np.ascontiguousarray(
        query_embedding.astype(np.float32).reshape(1, -1))
    faiss.normalize_L2(query)                               #F
    scores, ids = index.search(query, k)                    #G
    return ids[0], scores[0]

#A Normalize so the inner product equals cosine similarity
#B An HNSW index, 32 connections per node -- the metric is explicit because the default is L2
#C Higher efConstruction improves recall while building the graph
#D efSearch must be >= k; FAISS silently raises it to k when violated, so a value below k gives no speed benefit
#E Add every item embedding to the index
#F The query must be normalized exactly like the index
#G Top-k nearest FAISS positions and their similarity scores


# ---------------------------------------------------------------------------
# Production-shaped index
# ---------------------------------------------------------------------------

class ANNRetrievalIndex:
    """FAISS-backed retrieval with ID mapping, incremental adds, and recall checks.

    Wraps FAISS with the pieces a production system needs and a raw index
    lacks: a bidirectional mapping between FAISS positions and application
    item IDs, incremental additions for items published after the index was
    built, and recall measurement against exact search -- the acceptance
    test to run before deploying any index configuration (0.95+ against
    exact search is typically acceptable).

    FAISS does not support removing vectors from most index types; deleted
    or unpublished items are filtered *after* retrieval, which is cheap
    because the candidate pool is only a few hundred items.
    """

    def __init__(self, index_type: str = "hnsw", emb_dim: int = 64,
                 hnsw_m: int = 32, ef_construction: int = 200,
                 ef_search: int = 500, ivf_nlist: int = 100,
                 ivf_nprobe: int = 10):
        self.index_type = index_type
        self.emb_dim = emb_dim
        self.faiss_to_item: list = []                       #A
        self.item_to_faiss: dict = {}                       #A
        self._embeddings: list = []
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(emb_dim)         #B
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                emb_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(emb_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, emb_dim, ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = ivf_nprobe                  #C
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)
        return vectors

    def add_items(self, item_ids: list, embeddings: np.ndarray) -> None:
        vectors = self._normalize(embeddings)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(vectors)                       #D
        for item_id in item_ids:
            self.item_to_faiss[item_id] = len(self.faiss_to_item)
            self.faiss_to_item.append(item_id)              #E
        self.index.add(vectors)
        self._embeddings.append(vectors)

    def retrieve(self, query_embedding: np.ndarray, k: int = 500,
                 exclude: Optional[set] = None) -> list:
        """Top-k (item_id, score) pairs, excluding ``exclude`` after search."""
        query = self._normalize(query_embedding.reshape(1, -1))
        exclude = exclude or set()
        scores, ids = self.index.search(query, k + len(exclude))
        results = []
        for faiss_id, score in zip(ids[0], scores[0]):
            if faiss_id == -1:
                break
            item_id = self.faiss_to_item[int(faiss_id)]
            if item_id in exclude:                          #F
                continue
            results.append((item_id, float(score)))
            if len(results) == k:
                break
        return results

    def recall_vs_exact(self, queries: np.ndarray, k: int = 100) -> float:
        """Fraction of exact top-k neighbors the ANN index also returns."""
        all_vectors = np.vstack(self._embeddings)
        exact = faiss.IndexFlatIP(self.emb_dim)             #G
        exact.add(all_vectors)
        q = self._normalize(queries)
        _, exact_ids = exact.search(q, k)
        _, ann_ids = self.index.search(q, k)
        hits = sum(len(set(e) & set(a))
                   for e, a in zip(exact_ids, ann_ids))
        return hits / (len(queries) * k)                    #H

#A The bidirectional mapping between FAISS positions and item IDs
#B A flat index is exact search -- the baseline every ANN config is measured against
#C nprobe = how many of the nlist clusters to visit per query
#D IVF must learn its k-means clusters before vectors can be added
#E New items are appended, never inserted -- FAISS positions are stable
#F Deleted or already-seen items are filtered after retrieval
#G Exact search is the ground truth for ANN recall
#H ANN recall = overlap between approximate and exact top-k


# ---------------------------------------------------------------------------
# The retrieval stage for the four-stage pipeline
# ---------------------------------------------------------------------------

class ANNRetrieval(Retrieval):
    """ANN-backed retrieval stage.

    Same interface as the Chapter 2 retrieval stages, milliseconds instead
    of seconds. Seed item IDs are looked up in the query-tower table; with
    several seeds, their query embeddings are averaged. For user-seeded
    (U2I) retrieval, replace the seed lookup with the user-tower embedding
    -- the index does not care where the query vector comes from.
    """

    def __init__(self, ann_index: ANNRetrievalIndex,
                 query_embeddings: np.ndarray,
                 item_to_idx: dict):
        self.ann_index = ann_index                          #A
        self.query_embeddings = query_embeddings
        self.item_to_idx = item_to_idx
        self.num_items = query_embeddings.shape[0]

    def retrieve_similar_items(self, seed_ids: list, k: int = 100) -> List[ScoredItem]:
        seeds = [s for s in (seed_ids or []) if s in self.item_to_idx]
        if not seeds:
            return []
        vectors = self.query_embeddings[[self.item_to_idx[s] for s in seeds]]
        query = vectors.mean(axis=0)                        #B
        results = self.ann_index.retrieve(query, k=k, exclude=set(seeds))  #C
        return [
            ScoredItem(item_id=item_id, scores={"similarity": score})  #D
            for item_id, score in results
        ]

    def retrieve(self, context: RecommendationContext) -> List[ScoredItem]:
        return self.retrieve_similar_items(
            context.seed_items, k=max(100, context.k))      #E

#A A prebuilt ANNRetrievalIndex over the candidate-tower embeddings
#B Multiple seeds are averaged into one query vector
#C The seeds themselves are excluded from their own candidate pool
#D Retrieval scores travel with the candidate through the pipeline
#E Retrieve a wide pool -- later stages narrow it down


# ---------------------------------------------------------------------------
# Cold start (Section 5.2.4)
# ---------------------------------------------------------------------------

def warm_start_embedding(new_item_genres: list,
                         genre_to_items: dict,
                         item_embeddings: np.ndarray) -> np.ndarray:
    """Initialize a new item's embedding from its genre neighbors.

    A brand-new item has no interactions, so the trained model cannot give
    it an embedding -- and an item without an embedding is invisible to ANN
    retrieval. Averaging the embeddings of existing items that share a genre
    is crude, but far better than invisible; the next training cycle
    replaces it with a behavioral embedding.
    """
    neighbor_indices = set()
    for genre in new_item_genres:
        neighbor_indices.update(genre_to_items.get(genre, []))   #A
    if not neighbor_indices:
        vec = np.random.randn(item_embeddings.shape[1])          #B
        return (vec / np.linalg.norm(vec)).astype(np.float32)
    neighbors = item_embeddings[list(neighbor_indices)]
    embedding = neighbors.mean(axis=0)                           #C
    return (embedding / np.linalg.norm(embedding)).astype(np.float32)  #D

#A Collect existing items that share any genre with the new item
#B Fall back to a random direction when no genre matches exist
#C The average of the genre neighbors is the initial representation
#D Normalize so the new vector matches the index
