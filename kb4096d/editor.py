"""KB editing operations: add, remove, move, interpolate, analogy, merge."""

from typing import List, Optional, Tuple

import torch

from .kb_store import KnowledgeBase
from .model_loader import ModelManagerProtocol
from .utils import (
    normalize_vector,
    interpolate_vectors,
    analogy_vector,
    cosine_sim,
    top_k_similar,
)


class KBEditor:
    """Editor for knowledge base operations — all in native vector space."""

    def __init__(self, model_mgr: ModelManagerProtocol):
        self.model_mgr = model_mgr

    def add_concept(
        self,
        kb: KnowledgeBase,
        name: str,
        text: Optional[str] = None,
        vector: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a concept to a KB. Uses model to extract vector from text if no vector given."""
        if vector is None:
            text = text or name
            vector = self.model_mgr.get_hidden(text)
        kb.add_concept(name, normalize_vector(vector))

    def add_concepts_batch(
        self,
        kb: KnowledgeBase,
        names: List[str],
        texts: Optional[List[str]] = None,
    ) -> None:
        """Add multiple concepts efficiently in a batch."""
        texts = texts or names
        vectors = self.model_mgr.get_hidden_batch(texts)
        for name, vec in zip(names, vectors):
            kb.add_concept(name, normalize_vector(vec))

    def remove_concept(self, kb: KnowledgeBase, name: str) -> bool:
        """Remove a concept and its relations."""
        return kb.remove_concept(name)

    def move_concept(
        self,
        kb: KnowledgeBase,
        name: str,
        direction_text: str,
        strength: float = 0.3,
    ) -> None:
        """Move a concept vector towards a direction (expressed as text).

        The concept is interpolated with the direction vector.
        strength=0.0 keeps original, strength=1.0 fully replaces with direction.
        """
        concept = kb.get_concept(name)
        if concept is None:
            raise KeyError(f"Concept '{name}' not found in KB.")

        direction_vec = self.model_mgr.get_hidden(direction_text)
        new_vec = interpolate_vectors(concept.vector, direction_vec, alpha=strength)
        kb.add_concept(name, new_vec, concept.metadata)

    def interpolate(
        self,
        kb: KnowledgeBase,
        new_name: str,
        concept_a: str,
        concept_b: str,
        alpha: float = 0.5,
    ) -> None:
        """Create a new concept by interpolating between two existing ones."""
        ca = kb.get_concept(concept_a)
        cb = kb.get_concept(concept_b)
        if ca is None:
            raise KeyError(f"Concept '{concept_a}' not found.")
        if cb is None:
            raise KeyError(f"Concept '{concept_b}' not found.")

        new_vec = interpolate_vectors(ca.vector, cb.vector, alpha=alpha)
        kb.add_concept(new_name, new_vec)

    def analogy(
        self,
        kb: KnowledgeBase,
        new_name: str,
        base: str,
        source: str,
        target: str,
    ) -> Tuple[torch.Tensor, List[Tuple[str, float]]]:
        """Create a concept by analogy: base + (target - source).

        E.g., analogy(kb, "capitale_brasile", base="Brasile", source="Francia", target="Parigi")
        → should land near "Brasilia"

        Returns:
            Tuple of (result_vector, nearest_neighbors_in_kb).
        """
        c_base = kb.get_concept(base)
        c_src = kb.get_concept(source)
        c_dst = kb.get_concept(target)
        if c_base is None:
            raise KeyError(f"Concept '{base}' not found.")
        if c_src is None:
            raise KeyError(f"Concept '{source}' not found.")
        if c_dst is None:
            raise KeyError(f"Concept '{target}' not found.")

        result_vec = analogy_vector(c_base.vector, c_src.vector, c_dst.vector)
        kb.add_concept(new_name, result_vec)

        # Find nearest neighbors for the result
        names = kb.get_concept_names()
        vectors = kb.get_concept_vectors()
        neighbors = top_k_similar(result_vec, names, vectors, k=5)

        return result_vec, neighbors

    def create_relation(
        self,
        kb: KnowledgeBase,
        source: str,
        relation_type: str,
        target: str,
    ) -> None:
        """Create an explicit relation between two concepts."""
        if kb.get_concept(source) is None:
            raise KeyError(f"Source concept '{source}' not found.")
        if kb.get_concept(target) is None:
            raise KeyError(f"Target concept '{target}' not found.")

        rel_text = f"{source} {relation_type} {target}"
        rel_vec = self.model_mgr.get_hidden(rel_text)
        kb.add_relation(source, relation_type, target, normalize_vector(rel_vec))

    def correct_fact(
        self,
        kb: KnowledgeBase,
        concept_name: str,
        new_text: str,
    ) -> None:
        """Replace a concept's vector with one derived from new text.

        Useful for correcting facts: the old vector is replaced entirely.
        """
        concept = kb.get_concept(concept_name)
        if concept is None:
            raise KeyError(f"Concept '{concept_name}' not found.")

        new_vec = self.model_mgr.get_hidden(new_text)
        kb.add_concept(concept_name, normalize_vector(new_vec), concept.metadata)

    def merge_kbs(
        self,
        target: KnowledgeBase,
        source: KnowledgeBase,
        conflict_strategy: str = "keep_target",
    ) -> int:
        """Merge source KB into target KB.

        Args:
            target: KB to merge into (modified in place).
            source: KB to merge from (not modified).
            conflict_strategy: "keep_target", "keep_source", or "average".

        Returns:
            Number of concepts added or modified.
        """
        changes = 0
        for name, concept in source.concepts.items():
            if name in target.concepts:
                if conflict_strategy == "keep_target":
                    continue
                elif conflict_strategy == "keep_source":
                    target.add_concept(name, concept.vector, concept.metadata)
                    changes += 1
                elif conflict_strategy == "average":
                    existing = target.concepts[name]
                    avg_vec = interpolate_vectors(existing.vector, concept.vector, 0.5)
                    merged_meta = {**existing.metadata, **concept.metadata}
                    target.add_concept(name, avg_vec, merged_meta)
                    changes += 1
            else:
                target.add_concept(name, concept.vector, concept.metadata)
                changes += 1

        # Merge relations (avoid duplicates)
        existing_rels = {
            (r.source, r.relation_type, r.target) for r in target.relations
        }
        for rel in source.relations:
            key = (rel.source, rel.relation_type, rel.target)
            if key not in existing_rels:
                target.add_relation(
                    rel.source, rel.relation_type, rel.target,
                    rel.vector, rel.metadata,
                )
                changes += 1

        return changes
