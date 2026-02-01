"""Knowledge Base data structure: save/load as .pt files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch


@dataclass
class Concept:
    """A single concept in the KB: a name and its vector representation."""
    name: str
    vector: torch.Tensor  # shape (D,), stored on CPU
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Relation:
    """A directed relation between two concepts."""
    source: str
    relation_type: str
    target: str
    vector: torch.Tensor  # shape (D,), the relation's own embedding
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Override:
    """A vector override that modifies model behavior at a specific layer."""
    name: str
    vector: torch.Tensor  # shape (D,)
    layer: int
    strength: float = 1.0


class KnowledgeBase:
    """A single knowledge base containing concepts, relations, and overrides.

    All vectors live on CPU. The model stays on GPU. This separation prevents
    VRAM exhaustion with large KBs.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        hidden_dim: Optional[int] = None,
        model_name: str = "",
    ):
        self.name = name
        self.description = description
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.created_at = datetime.now().isoformat()
        self.modified_at = self.created_at

        self.concepts: Dict[str, Concept] = {}
        self.relations: list[Relation] = []
        self.overrides: Dict[str, Override] = {}

    def add_concept(self, name: str, vector: torch.Tensor, metadata: Optional[Dict[str, str]] = None) -> None:
        """Add or update a concept."""
        self.concepts[name] = Concept(
            name=name,
            vector=vector.cpu().float(),
            metadata=metadata or {},
        )
        self._touch()

    def remove_concept(self, name: str) -> bool:
        """Remove a concept and its relations. Returns True if found."""
        if name not in self.concepts:
            return False
        del self.concepts[name]
        self.relations = [
            r for r in self.relations
            if r.source != name and r.target != name
        ]
        self.overrides.pop(name, None)
        self._touch()
        return True

    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        vector: torch.Tensor,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a relation between two concepts."""
        self.relations.append(Relation(
            source=source,
            relation_type=relation_type,
            target=target,
            vector=vector.cpu().float(),
            metadata=metadata or {},
        ))
        self._touch()

    def add_override(self, name: str, vector: torch.Tensor, layer: int, strength: float = 1.0) -> None:
        """Add a vector override for a specific layer."""
        self.overrides[name] = Override(
            name=name,
            vector=vector.cpu().float(),
            layer=layer,
            strength=strength,
        )
        self._touch()

    def get_concept_names(self) -> list[str]:
        return list(self.concepts.keys())

    def get_concept_vectors(self) -> torch.Tensor:
        """Return stacked concept vectors as (N, D) tensor."""
        if not self.concepts:
            dim = self.hidden_dim or 1
            return torch.zeros(0, dim)
        return torch.stack([c.vector for c in self.concepts.values()])

    def get_concept(self, name: str) -> Optional[Concept]:
        return self.concepts.get(name)

    def get_relations_for(self, concept_name: str) -> list[Relation]:
        """Get all relations involving a concept (as source or target)."""
        return [
            r for r in self.relations
            if r.source == concept_name or r.target == concept_name
        ]

    def get_relations_by_type(self, relation_type: str) -> list[Relation]:
        return [r for r in self.relations if r.relation_type == relation_type]

    @property
    def centroid(self) -> Optional[torch.Tensor]:
        """Compute the centroid of all concept vectors. Used for routing."""
        if not self.concepts:
            return None
        vectors = self.get_concept_vectors()
        return vectors.mean(dim=0)

    def _touch(self):
        self.modified_at = datetime.now().isoformat()

    def save(self, path: Path) -> None:
        """Save KB to a .pt file."""
        path = Path(path)
        data = {
            "name": self.name,
            "description": self.description,
            "hidden_dim": self.hidden_dim,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "concepts": {
                name: {
                    "vector": c.vector,
                    "metadata": c.metadata,
                }
                for name, c in self.concepts.items()
            },
            "relations": [
                {
                    "source": r.source,
                    "relation_type": r.relation_type,
                    "target": r.target,
                    "vector": r.vector,
                    "metadata": r.metadata,
                }
                for r in self.relations
            ],
            "overrides": {
                name: {
                    "vector": o.vector,
                    "layer": o.layer,
                    "strength": o.strength,
                }
                for name, o in self.overrides.items()
            },
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        """Load KB from a .pt file."""
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)

        kb = cls(
            name=data["name"],
            description=data.get("description", ""),
            hidden_dim=data.get("hidden_dim"),
            model_name=data.get("model_name", ""),
        )
        kb.created_at = data.get("created_at", kb.created_at)
        kb.modified_at = data.get("modified_at", kb.modified_at)

        for name, cdata in data.get("concepts", {}).items():
            kb.concepts[name] = Concept(
                name=name,
                vector=cdata["vector"].cpu().float(),
                metadata=cdata.get("metadata", {}),
            )

        for rdata in data.get("relations", []):
            kb.relations.append(Relation(
                source=rdata["source"],
                relation_type=rdata["relation_type"],
                target=rdata["target"],
                vector=rdata["vector"].cpu().float(),
                metadata=rdata.get("metadata", {}),
            ))

        for name, odata in data.get("overrides", {}).items():
            kb.overrides[name] = Override(
                name=name,
                vector=odata["vector"].cpu().float(),
                layer=odata["layer"],
                strength=odata.get("strength", 1.0),
            )

        return kb

    def __repr__(self) -> str:
        return (
            f"KnowledgeBase(name={self.name!r}, "
            f"concepts={len(self.concepts)}, "
            f"relations={len(self.relations)}, "
            f"dim={self.hidden_dim})"
        )
