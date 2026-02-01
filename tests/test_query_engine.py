"""Tests for QueryEngine: search, relate, chain."""

import torch
import pytest

from kb4096d.config import KBConfig
from kb4096d.query_engine import QueryEngine
from kb4096d.kb_store import KnowledgeBase


class TestQueryEngine:
    def test_search_returns_results(self, mock_model, config, populated_kb):
        engine = QueryEngine(mock_model, config)
        # Use threshold=0.0 because mock vectors are random (low similarity expected)
        results = engine.search("capitale del Giappone", populated_kb, threshold=0.0)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_search_empty_kb(self, mock_model, config):
        engine = QueryEngine(mock_model, config)
        kb = KnowledgeBase("empty", hidden_dim=mock_model.hidden_dim)
        results = engine.search("anything", kb)
        assert results == []

    def test_search_by_vector(self, mock_model, config, populated_kb):
        engine = QueryEngine(mock_model, config)
        query_vec = mock_model.get_hidden("Tokyo")
        results = engine.search_by_vector(query_vec, populated_kb)
        assert len(results) > 0
        # The exact vector for "Tokyo" should match itself perfectly
        top_name, top_sim = results[0]
        assert top_name == "Tokyo"
        assert top_sim > 0.99

    def test_search_respects_top_k(self, mock_model, config, populated_kb):
        engine = QueryEngine(mock_model, config)
        results = engine.search("test", populated_kb, top_k=2)
        assert len(results) <= 2

    def test_search_respects_threshold(self, mock_model, config, populated_kb):
        engine = QueryEngine(mock_model, config)
        results = engine.search("test", populated_kb, threshold=0.99)
        # With high threshold, most results should be filtered
        for name, sim in results:
            assert sim >= 0.99

    def test_relate_returns_results(self, mock_model, config, populated_kb):
        # Lower threshold for mock vectors (random, low cosine similarity)
        config.similarity_threshold = 0.0
        engine = QueryEngine(mock_model, config)
        results = engine.relate("Giappone", "capitale_di", populated_kb)
        assert len(results) > 0

    def test_relate_excludes_self(self, mock_model, config, populated_kb):
        config.similarity_threshold = 0.0
        engine = QueryEngine(mock_model, config)
        results = engine.relate("Giappone", "capitale_di", populated_kb)
        names = [r[0] for r in results]
        assert "Giappone" not in names

    def test_relate_with_explicit_relation(self, mock_model, config, populated_kb):
        """When explicit relations exist, they should be returned."""
        from kb4096d.extractor import extract_relation
        rel_vec = extract_relation(mock_model, "Giappone", "capitale_di", "Tokyo")
        populated_kb.add_relation("Giappone", "capitale_di", "Tokyo", rel_vec)

        engine = QueryEngine(mock_model, config)
        results = engine.relate("Giappone", "capitale_di", populated_kb)
        assert len(results) > 0
        assert results[0][0] == "Tokyo"

    def test_chain(self, mock_model, config, populated_kb):
        engine = QueryEngine(mock_model, config)
        chain_results = engine.chain("Italia", ["contiene", "capitale"], populated_kb)
        assert len(chain_results) == 2
        for step in chain_results:
            assert isinstance(step, list)

    def test_multi_kb_search(self, mock_model, config, populated_kb):
        config.similarity_threshold = 0.0
        engine = QueryEngine(mock_model, config)
        kb2 = KnowledgeBase("kb2", hidden_dim=mock_model.hidden_dim)
        kb2.add_concept("test_concept", mock_model.get_hidden("test_concept"))

        results = engine.multi_kb_search("test", [populated_kb, kb2])
        assert len(results) > 0
        # Results should include kb_name
        assert all(len(r) == 3 for r in results)
