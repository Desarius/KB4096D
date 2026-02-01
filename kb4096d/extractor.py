"""Concept and relation extraction from the model's hidden space."""

from typing import List, Optional, Tuple

import torch

from .config import KBConfig
from .kb_store import KnowledgeBase
from .model_loader import ModelManagerProtocol
from .utils import normalize_vector


def extract_concepts(
    model_mgr: ModelManagerProtocol,
    concept_names: List[str],
) -> List[Tuple[str, torch.Tensor]]:
    """Extract concept vectors from the model.

    Each concept name is passed through the model and its hidden state
    is captured at the extraction layer. This is the ONLY point where
    text becomes a vector.

    Returns:
        List of (name, vector) tuples.
    """
    vectors = model_mgr.get_hidden_batch(concept_names)  # (N, D)
    results = []
    for name, vec in zip(concept_names, vectors):
        results.append((name, normalize_vector(vec)))
    return results


def extract_relation(
    model_mgr: ModelManagerProtocol,
    source: str,
    relation_type: str,
    target: str,
) -> torch.Tensor:
    """Extract a relation vector by encoding the relationship phrase.

    The relation is encoded as a sentence expressing the relationship,
    giving the model context about how source and target relate.

    Returns:
        Normalized vector of shape (D,).
    """
    # Encode the relationship as a natural language statement
    relation_text = f"{source} {relation_type} {target}"
    vec = model_mgr.get_hidden(relation_text)
    return normalize_vector(vec)


def build_kb(
    model_mgr: ModelManagerProtocol,
    name: str,
    description: str,
    concept_names: List[str],
    relations: Optional[List[Tuple[str, str, str]]] = None,
) -> KnowledgeBase:
    """Build a complete KB from concept names and optional relations.

    Args:
        model_mgr: The model manager for vector extraction.
        name: KB name.
        description: KB description.
        concept_names: List of concept strings to extract.
        relations: Optional list of (source, relation_type, target) triples.

    Returns:
        Populated KnowledgeBase.
    """
    kb = KnowledgeBase(
        name=name,
        description=description,
        hidden_dim=model_mgr.hidden_dim,
        model_name=getattr(model_mgr, 'model_name', ''),
    )

    # Extract concepts
    concept_data = extract_concepts(model_mgr, concept_names)
    for cname, vec in concept_data:
        kb.add_concept(cname, vec)

    # Extract relations
    if relations:
        for source, rel_type, target in relations:
            rel_vec = extract_relation(model_mgr, source, rel_type, target)
            kb.add_relation(source, rel_type, target, rel_vec)

    return kb
