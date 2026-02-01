"""Search, relate, and chain queries in native vector space."""

from typing import List, Optional, Tuple

import torch

from .config import KBConfig
from .kb_store import KnowledgeBase
from .model_loader import ModelManagerProtocol
from .utils import cosine_sim, cosine_sim_batch, top_k_similar, normalize_vector


class QueryEngine:
    """Query engine operating entirely in the model's native vector space.

    Text exists only at the border: the query text is converted to a vector
    once via get_hidden(), then all operations are pure vector algebra.
    """

    def __init__(self, model_mgr: ModelManagerProtocol, config: KBConfig):
        self.model_mgr = model_mgr
        self.config = config

    def search(
        self,
        query_text: str,
        kb: KnowledgeBase,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Search a KB for concepts similar to a query text.

        The query is converted to a vector ONCE, then compared against
        all concept vectors using cosine similarity.

        Returns:
            List of (concept_name, similarity) tuples.
        """
        top_k = top_k or self.config.default_top_k
        threshold = threshold if threshold is not None else self.config.similarity_threshold

        query_vec = self.model_mgr.get_hidden(query_text)
        return self.search_by_vector(query_vec, kb, top_k, threshold)

    def search_by_vector(
        self,
        query_vec: torch.Tensor,
        kb: KnowledgeBase,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Search a KB using a pre-computed vector."""
        top_k = top_k or self.config.default_top_k
        threshold = threshold if threshold is not None else self.config.similarity_threshold

        names = kb.get_concept_names()
        vectors = kb.get_concept_vectors()

        if vectors.size(0) == 0:
            return []

        return top_k_similar(query_vec, names, vectors, k=top_k, threshold=threshold)

    def relate(
        self,
        concept_name: str,
        relation_type: str,
        kb: KnowledgeBase,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Find concepts related to a given concept via a relation type.

        Strategy: encode "{concept} {relation_type}" as a query vector,
        then search the KB for the closest concepts (excluding the source).

        Returns:
            List of (concept_name, similarity).
        """
        top_k = top_k or self.config.default_top_k

        # Build query from concept + relation
        query_text = f"{concept_name} {relation_type}"
        query_vec = self.model_mgr.get_hidden(query_text)

        # Also check if there are explicit relations in the KB
        explicit = self._find_explicit_relations(concept_name, relation_type, kb)
        if explicit:
            return explicit

        # Fall back to vector search, excluding the source concept
        names = kb.get_concept_names()
        vectors = kb.get_concept_vectors()

        if vectors.size(0) == 0:
            return []

        results = top_k_similar(
            query_vec, names, vectors,
            k=top_k + 1,  # +1 to account for possible self-match
            threshold=self.config.similarity_threshold,
        )

        # Filter out the source concept itself
        results = [(n, s) for n, s in results if n != concept_name]
        return results[:top_k]

    def chain(
        self,
        start_concept: str,
        relation_types: List[str],
        kb: KnowledgeBase,
    ) -> List[List[Tuple[str, float]]]:
        """Follow a chain of relations from a starting concept.

        E.g., chain("Europa", ["contiene", "capitale_di"]) might yield:
        Step 1: Europa → [Italia, Francia, ...]
        Step 2: Italia → [Roma], Francia → [Parigi], ...

        Returns:
            List of result lists, one per relation step.
        """
        current_concepts = [start_concept]
        chain_results = []

        for rel_type in relation_types:
            step_results = []
            for concept in current_concepts:
                results = self.relate(concept, rel_type, kb, top_k=3)
                step_results.extend(results)

            # Deduplicate by name, keeping highest similarity
            seen = {}
            for name, sim in step_results:
                if name not in seen or sim > seen[name]:
                    seen[name] = sim
            deduped = sorted(seen.items(), key=lambda x: x[1], reverse=True)

            chain_results.append(deduped)
            current_concepts = [name for name, _ in deduped[:3]]

        return chain_results

    def multi_kb_search(
        self,
        query_text: str,
        kbs: List[KnowledgeBase],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Search across multiple KBs.

        Returns:
            List of (kb_name, concept_name, similarity).
        """
        top_k = top_k or self.config.default_top_k
        query_vec = self.model_mgr.get_hidden(query_text)

        all_results = []
        for kb in kbs:
            results = self.search_by_vector(query_vec, kb, top_k=top_k)
            for name, sim in results:
                all_results.append((kb.name, name, sim))

        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:top_k]

    def _find_explicit_relations(
        self,
        concept_name: str,
        relation_type: str,
        kb: KnowledgeBase,
    ) -> List[Tuple[str, float]]:
        """Check for explicitly stored relations."""
        results = []
        for rel in kb.relations:
            if rel.source == concept_name and rel.relation_type == relation_type:
                target_concept = kb.get_concept(rel.target)
                if target_concept is not None:
                    sim = cosine_sim(rel.vector, target_concept.vector)
                    results.append((rel.target, sim))
            elif rel.target == concept_name and rel.relation_type == relation_type:
                source_concept = kb.get_concept(rel.source)
                if source_concept is not None:
                    sim = cosine_sim(rel.vector, source_concept.vector)
                    results.append((rel.source, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)
