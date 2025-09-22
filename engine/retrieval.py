"""
Retrieval utilities for Curiosity AI.

This module provides a simple wrapper around a FAISS index built from a
collection of text snippets.  It also loads a corresponding metadata file
containing the raw texts and, optionally, a knowledge graph for more
sophisticated reasoning.  The primary method `search` returns the most
similar snippets to a given query.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import faiss
import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORTS", "1")
from sentence_transformers import SentenceTransformer
import networkx as nx


class Retriever:
    """Vector retrieval using FAISS and sentence transformers."""

    def __init__(self, embedder_name: str, index_path: str, graph_path: str | None = None) -> None:
        # Load the embedding model once.  SentenceTransformer caches models
        # locally and downloads them if necessary.  The model must match
        # the one used to build the FAISS index.
        self.model = SentenceTransformer(embedder_name)
        self.index = None
        self.texts: List[str] = []
        # Load the FAISS index.  A corresponding .meta.json file should exist
        # containing a dictionary with a "texts" field storing the original
        # sentences used to build the index.
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                meta_path = index_path + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.texts = meta.get("texts", [])
            except Exception:
                # If loading fails leave index as None.  Searches will return empty.
                self.index = None
        # Load an optional knowledge graph.  Not currently used by the engine
        # but may be useful for extensions.  If the file does not exist the
        # graph attribute will be an empty graph.
        if graph_path and os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                except Exception:
                    self.graph = nx.Graph()
        else:
            self.graph = nx.Graph()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search the FAISS index for the `k` nearest neighbours of `query`.

        Parameters
        ----------
        query : str
            The query string to embed and search for similar sentences.
        k : int, optional
            Number of results to return.  Defaults to 5.

        Returns
        -------
        List[Tuple[str, float]]
            A list of tuples `(text, distance)` where `text` is a sentence
            from the corpus and `distance` is the squared Euclidean distance
            between the query embedding and the sentence embedding.  If the
            index is not available an empty list is returned.
        """
        if self.index is None or not self.texts:
            return []
        # Compute the embedding for the query.  FAISS expects float32 arrays.
        emb = self.model.encode([query])
        emb = np.array(emb, dtype="float32")
        # Perform the search.  The FAISS index stores squared L2 distances by default.
        distances, indices = self.index.search(emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.texts[idx], float(dist)))
        return results