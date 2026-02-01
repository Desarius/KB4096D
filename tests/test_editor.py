"""Tests for KBEditor: add, remove, move, interpolate, analogy, merge."""

import torch
import pytest

from kb4096d.editor import KBEditor
from kb4096d.kb_store import KnowledgeBase
from kb4096d.utils import cosine_sim


class TestKBEditor:
    def test_add_concept(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        editor.add_concept(populated_kb, "Brasile")
        assert "Brasile" in populated_kb.concepts

    def test_add_concept_with_custom_text(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        editor.add_concept(populated_kb, "BRA", text="Il Brasile, paese del Sudamerica")
        assert "BRA" in populated_kb.concepts

    def test_add_concepts_batch(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        editor.add_concepts_batch(populated_kb, ["Brasile", "Brasilia", "Argentina"])
        assert "Brasile" in populated_kb.concepts
        assert "Brasilia" in populated_kb.concepts
        assert "Argentina" in populated_kb.concepts

    def test_remove_concept(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        assert editor.remove_concept(populated_kb, "Francia") is True
        assert "Francia" not in populated_kb.concepts
        assert editor.remove_concept(populated_kb, "NonExistent") is False

    def test_move_concept(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        original_vec = populated_kb.concepts["Francia"].vector.clone()
        editor.move_concept(populated_kb, "Francia", "Europa", strength=0.5)
        new_vec = populated_kb.concepts["Francia"].vector
        # Should have changed
        sim = cosine_sim(original_vec, new_vec)
        assert sim < 0.99  # Should be different

    def test_move_concept_not_found(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        with pytest.raises(KeyError):
            editor.move_concept(populated_kb, "NonExistent", "direction")

    def test_interpolate(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        editor.interpolate(populated_kb, "FrancItalia", "Francia", "Italia", alpha=0.5)
        assert "FrancItalia" in populated_kb.concepts
        vec = populated_kb.concepts["FrancItalia"].vector
        # Should be somewhere between Francia and Italia
        sim_fr = cosine_sim(vec, populated_kb.concepts["Francia"].vector)
        sim_it = cosine_sim(vec, populated_kb.concepts["Italia"].vector)
        assert sim_fr > 0 and sim_it > 0

    def test_interpolate_not_found(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        with pytest.raises(KeyError):
            editor.interpolate(populated_kb, "new", "NonExistent", "Francia")

    def test_analogy(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        vec, neighbors = editor.analogy(
            populated_kb, "capitale_giappone",
            base="Giappone", source="Francia", target="Parigi"
        )
        assert "capitale_giappone" in populated_kb.concepts
        assert len(neighbors) > 0

    def test_analogy_not_found(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        with pytest.raises(KeyError):
            editor.analogy(populated_kb, "x", "NonExistent", "Francia", "Parigi")

    def test_create_relation(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        editor.create_relation(populated_kb, "Francia", "capitale_di", "Parigi")
        rels = populated_kb.get_relations_by_type("capitale_di")
        assert len(rels) == 1
        assert rels[0].source == "Francia"
        assert rels[0].target == "Parigi"

    def test_create_relation_not_found(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        with pytest.raises(KeyError):
            editor.create_relation(populated_kb, "NonExistent", "rel", "Parigi")

    def test_correct_fact(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        original = populated_kb.concepts["Parigi"].vector.clone()
        editor.correct_fact(populated_kb, "Parigi", "Paris, capital of France")
        new = populated_kb.concepts["Parigi"].vector
        sim = cosine_sim(original, new)
        assert sim < 0.99  # Should be different

    def test_correct_fact_not_found(self, mock_model, populated_kb):
        editor = KBEditor(mock_model)
        with pytest.raises(KeyError):
            editor.correct_fact(populated_kb, "NonExistent", "new text")

    def test_merge_kbs(self, mock_model):
        editor = KBEditor(mock_model)

        target = KnowledgeBase("target", hidden_dim=mock_model.hidden_dim)
        target.add_concept("a", torch.randn(mock_model.hidden_dim))
        target.add_concept("shared", torch.randn(mock_model.hidden_dim))

        source = KnowledgeBase("source", hidden_dim=mock_model.hidden_dim)
        source.add_concept("b", torch.randn(mock_model.hidden_dim))
        source.add_concept("shared", torch.randn(mock_model.hidden_dim))

        changes = editor.merge_kbs(target, source, conflict_strategy="keep_target")
        assert "a" in target.concepts
        assert "b" in target.concepts
        assert "shared" in target.concepts
        assert changes == 1  # Only "b" was new

    def test_merge_kbs_keep_source(self, mock_model):
        editor = KBEditor(mock_model)
        dim = mock_model.hidden_dim

        target = KnowledgeBase("target", hidden_dim=dim)
        target.add_concept("shared", torch.ones(dim))

        source = KnowledgeBase("source", hidden_dim=dim)
        source.add_concept("shared", torch.zeros(dim))

        editor.merge_kbs(target, source, conflict_strategy="keep_source")
        assert torch.allclose(target.concepts["shared"].vector, torch.zeros(dim))

    def test_merge_kbs_average(self, mock_model):
        editor = KBEditor(mock_model)
        dim = mock_model.hidden_dim

        target = KnowledgeBase("target", hidden_dim=dim)
        target.add_concept("shared", torch.ones(dim))

        source = KnowledgeBase("source", hidden_dim=dim)
        source.add_concept("shared", torch.zeros(dim))

        editor.merge_kbs(target, source, conflict_strategy="average")
        vec = target.concepts["shared"].vector
        # Interpolated and normalized, so won't be exactly 0.5 but should be in between
        assert vec.mean().item() > -0.5 and vec.mean().item() < 1.0
