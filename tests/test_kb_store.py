"""Tests for KnowledgeBase save/load and operations."""

import torch
import pytest

from kb4096d.kb_store import KnowledgeBase


class TestKnowledgeBase:
    def test_create_empty(self):
        kb = KnowledgeBase("test", "desc", hidden_dim=256)
        assert kb.name == "test"
        assert kb.description == "desc"
        assert kb.hidden_dim == 256
        assert len(kb.concepts) == 0
        assert len(kb.relations) == 0

    def test_add_concept(self):
        kb = KnowledgeBase("test", hidden_dim=256)
        vec = torch.randn(256)
        kb.add_concept("foo", vec)
        assert "foo" in kb.concepts
        assert kb.concepts["foo"].vector.shape == (256,)

    def test_remove_concept(self):
        kb = KnowledgeBase("test", hidden_dim=256)
        kb.add_concept("foo", torch.randn(256))
        assert kb.remove_concept("foo") is True
        assert "foo" not in kb.concepts
        assert kb.remove_concept("foo") is False  # already removed

    def test_remove_concept_cleans_relations(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        kb.add_concept("a", torch.randn(4))
        kb.add_concept("b", torch.randn(4))
        kb.add_relation("a", "rel", "b", torch.randn(4))
        assert len(kb.relations) == 1
        kb.remove_concept("a")
        assert len(kb.relations) == 0

    def test_add_relation(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        kb.add_relation("a", "likes", "b", torch.randn(4))
        assert len(kb.relations) == 1
        assert kb.relations[0].source == "a"
        assert kb.relations[0].relation_type == "likes"
        assert kb.relations[0].target == "b"

    def test_get_concept_vectors(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        kb.add_concept("a", torch.ones(4))
        kb.add_concept("b", torch.zeros(4))
        vecs = kb.get_concept_vectors()
        assert vecs.shape == (2, 4)

    def test_get_concept_vectors_empty(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        vecs = kb.get_concept_vectors()
        assert vecs.shape == (0, 4)

    def test_centroid(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        kb.add_concept("a", torch.tensor([1.0, 0.0, 0.0, 0.0]))
        kb.add_concept("b", torch.tensor([0.0, 1.0, 0.0, 0.0]))
        centroid = kb.centroid
        expected = torch.tensor([0.5, 0.5, 0.0, 0.0])
        assert torch.allclose(centroid, expected)

    def test_centroid_empty(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        assert kb.centroid is None

    def test_save_load(self, tmp_path):
        kb = KnowledgeBase("saveme", "save test", hidden_dim=4, model_name="test-model")
        kb.add_concept("a", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        kb.add_concept("b", torch.tensor([5.0, 6.0, 7.0, 8.0]))
        kb.add_relation("a", "rel", "b", torch.tensor([0.1, 0.2, 0.3, 0.4]))
        kb.add_override("override1", torch.randn(4), layer=5, strength=0.8)

        path = tmp_path / "test.pt"
        kb.save(path)
        assert path.exists()

        loaded = KnowledgeBase.load(path)
        assert loaded.name == "saveme"
        assert loaded.description == "save test"
        assert loaded.hidden_dim == 4
        assert loaded.model_name == "test-model"
        assert len(loaded.concepts) == 2
        assert torch.allclose(loaded.concepts["a"].vector, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert len(loaded.relations) == 1
        assert loaded.relations[0].relation_type == "rel"
        assert len(loaded.overrides) == 1
        assert loaded.overrides["override1"].layer == 5
        assert loaded.overrides["override1"].strength == 0.8

    def test_get_relations_for(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        kb.add_relation("a", "r1", "b", torch.randn(4))
        kb.add_relation("b", "r2", "c", torch.randn(4))
        rels = kb.get_relations_for("b")
        assert len(rels) == 2

    def test_get_relations_by_type(self):
        kb = KnowledgeBase("test", hidden_dim=4)
        kb.add_relation("a", "likes", "b", torch.randn(4))
        kb.add_relation("c", "hates", "d", torch.randn(4))
        assert len(kb.get_relations_by_type("likes")) == 1
        assert len(kb.get_relations_by_type("hates")) == 1
        assert len(kb.get_relations_by_type("loves")) == 0

    def test_repr(self):
        kb = KnowledgeBase("test", hidden_dim=256)
        r = repr(kb)
        assert "test" in r
        assert "256" in r
