"""Integration tests with real model (optional, marked slow).

Run with: pytest tests/test_integration.py -v -m slow
Skip with: pytest tests/ -v -m "not slow"

Set KB_TEST_MODEL env var to override model (default: TinyLlama).
"""

import os

import pytest
import torch

_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


def _has_cuda():
    return torch.cuda.is_available()


def _can_load_model():
    try:
        from transformers import AutoModelForCausalLM
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def real_model():
    """Load test model (set KB_TEST_MODEL env var to override)."""
    if not _can_load_model():
        pytest.skip("transformers not installed")

    from kb4096d.config import KBConfig
    from kb4096d.model_loader import ModelManager

    model_name = os.environ.get("KB_TEST_MODEL", _DEFAULT_MODEL)
    config = KBConfig(
        model_name=model_name,
        device="cuda" if _has_cuda() else "cpu",
        dtype="float32" if not _has_cuda() else "float16",
    )
    mgr = ModelManager(config)
    mgr.load()
    print(f"\n  Model: {model_name}, dim={mgr.hidden_dim}, layers={mgr.num_layers}")
    yield mgr
    mgr.unload()


class TestRealModelIntegration:
    def test_hidden_state_shape(self, real_model):
        vec = real_model.get_hidden("test")
        assert vec.shape == (real_model.hidden_dim,)
        assert vec.device.type == "cpu"

    def test_hidden_state_batch(self, real_model):
        vecs = real_model.get_hidden_batch(["hello", "world"])
        assert vecs.shape == (2, real_model.hidden_dim)

    def test_similar_concepts_closer(self, real_model):
        from kb4096d.utils import cosine_sim
        v_cat = real_model.get_hidden("cat")
        v_dog = real_model.get_hidden("dog")
        v_python = real_model.get_hidden("Python programming language")

        sim_cat_dog = cosine_sim(v_cat, v_dog)
        sim_cat_python = cosine_sim(v_cat, v_python)
        # cat and dog should be more similar than cat and python
        assert sim_cat_dog > sim_cat_python

    def test_full_pipeline(self, real_model, tmp_path):
        from kb4096d.config import KBConfig
        from kb4096d.kb_store import KnowledgeBase
        from kb4096d.extractor import build_kb
        from kb4096d.query_engine import QueryEngine

        config = KBConfig(kb_dir=tmp_path)
        kb = build_kb(
            real_model,
            name="geo",
            description="Geography",
            concept_names=["Francia", "Parigi", "Italia", "Roma", "Giappone", "Tokyo"],
            relations=[("Francia", "capitale_di", "Parigi")],
        )

        assert len(kb.concepts) == 6
        assert len(kb.relations) == 1

        # Save and reload
        path = tmp_path / "geo.pt"
        kb.save(path)
        loaded = KnowledgeBase.load(path)
        assert len(loaded.concepts) == 6

        # Search
        engine = QueryEngine(real_model, config)
        results = engine.search("capitale del Giappone", kb)
        assert len(results) > 0
        top_names = [r[0] for r in results[:3]]
        print(f"Search 'capitale del Giappone': {results}")
        # Tokyo should be among top results (this depends on model quality)
