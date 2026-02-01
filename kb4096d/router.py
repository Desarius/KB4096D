"""Automatic KB selection for queries via centroid comparison."""

from typing import List, Optional, Tuple

import torch

from .config import KBConfig
from .kb_manager import KBManager
from .model_loader import ModelManagerProtocol
from .utils import cosine_sim


class KBRouter:
    """Routes queries to the most relevant KB(s) automatically.

    Uses KB centroids (mean of all concept vectors) to quickly determine
    which KB is most relevant to a given query, without scanning all concepts.
    """

    def __init__(
        self,
        model_mgr: ModelManagerProtocol,
        kb_manager: KBManager,
        config: KBConfig,
    ):
        self.model_mgr = model_mgr
        self.kb_manager = kb_manager
        self.config = config

    def route(
        self,
        query_text: str,
        top_k: int = 1,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Find the most relevant KB(s) for a query.

        Args:
            query_text: The query to route.
            top_k: Number of KBs to return.
            threshold: Minimum centroid similarity to include.

        Returns:
            List of (kb_name, similarity) tuples.
        """
        query_vec = self.model_mgr.get_hidden(query_text)
        return self.route_by_vector(query_vec, top_k, threshold)

    def route_by_vector(
        self,
        query_vec: torch.Tensor,
        top_k: int = 1,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Route using a pre-computed vector."""
        results = self.kb_manager.find_relevant(query_vec, top_k=top_k)
        return [(name, sim) for name, sim in results if sim >= threshold]

    def auto_select(self, query_text: str) -> Optional[str]:
        """Select the single best KB for a query. Returns None if no good match."""
        results = self.route(query_text, top_k=1, threshold=self.config.similarity_threshold)
        if results:
            return results[0][0]
        return None
