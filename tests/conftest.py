"""Test fixtures: MockModelManager that works without GPU or real model."""

import hashlib
from typing import List

import pytest
import torch

from kb4096d.config import KBConfig


MOCK_DIM = 256


class MockModelManager:
    """Deterministic model manager for testing. No GPU, no real model.

    Produces consistent vectors from text by hashing the text and using
    the hash to seed a random generator. This gives deterministic but
    distinct vectors for different texts.
    """

    def __init__(self, hidden_dim: int = MOCK_DIM):
        self._hidden_dim = hidden_dim
        self._num_layers = 12
        self._extraction_layer = 12  # last layer (matches new default)
        self._is_loaded = True
        self.model_name = "mock-model"

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def extraction_layer(self) -> int:
        return self._extraction_layer

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _text_to_vector(self, text: str) -> torch.Tensor:
        """Deterministic text -> vector via hash seeding."""
        h = hashlib.sha256(text.encode()).hexdigest()
        seed = int(h[:8], 16)
        gen = torch.Generator()
        gen.manual_seed(seed)
        vec = torch.randn(self._hidden_dim, generator=gen)
        # L2 normalize
        return vec / vec.norm()

    def get_hidden(self, text: str) -> torch.Tensor:
        return self._text_to_vector(text)

    def get_hidden_batch(self, texts: List[str]) -> torch.Tensor:
        vecs = [self._text_to_vector(t) for t in texts]
        return torch.stack(vecs)

    def load(self):
        self._is_loaded = True

    def unload(self):
        self._is_loaded = False


@pytest.fixture
def mock_model():
    """Provide a MockModelManager."""
    return MockModelManager()


@pytest.fixture
def config(tmp_path):
    """Provide a KBConfig with temporary kb_dir."""
    return KBConfig(kb_dir=tmp_path / "kbs")


@pytest.fixture
def populated_kb(mock_model):
    """Provide a KB populated with geography concepts."""
    from kb4096d.kb_store import KnowledgeBase
    from kb4096d.extractor import extract_concepts

    kb = KnowledgeBase(
        name="geo_test",
        description="Test geography KB",
        hidden_dim=mock_model.hidden_dim,
        model_name="mock",
    )
    concepts = ["Francia", "Parigi", "Italia", "Roma", "Giappone", "Tokyo"]
    for name, vec in extract_concepts(mock_model, concepts):
        kb.add_concept(name, vec)
    return kb
