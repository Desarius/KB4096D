"""Deep semantic tests with a real model.

These tests verify that the system actually captures and manipulates
semantic knowledge, not just that the code runs without errors.

Categories:
1. Semantic discrimination: can the model tell apart unrelated concepts?
2. Relational structure: do country-capital pairs share geometric structure?
3. Analogy: does vector arithmetic produce semantically correct results?
4. Novel concepts: can we inject and retrieve invented knowledge?
5. Correction: can we overwrite a fact and see the change?
6. Cross-domain separation: are unrelated domains far apart?

Set KB_TEST_MODEL env var to override model (default: TinyLlama).
"""

import os

import pytest
import torch

_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pytestmark = pytest.mark.slow


def _has_cuda():
    return torch.cuda.is_available()


@pytest.fixture(scope="module")
def system():
    """Set up the full system once for all tests.

    Set KB_TEST_MODEL env var to choose model (default: TinyLlama).
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers not installed")

    from kb4096d.config import KBConfig
    from kb4096d.model_loader import ModelManager
    from kb4096d.kb_manager import KBManager
    from kb4096d.query_engine import QueryEngine
    from kb4096d.editor import KBEditor
    from kb4096d.router import KBRouter
    from kb4096d.utils import cosine_sim

    model_name = os.environ.get("KB_TEST_MODEL", _DEFAULT_MODEL)
    config = KBConfig(
        model_name=model_name,
        device="cuda" if _has_cuda() else "cpu",
        dtype="float16" if _has_cuda() else "float32",
        similarity_threshold=0.0,  # don't filter, we want to see everything
    )
    model = ModelManager(config)
    model.load()
    print(f"\n  Model: {model_name}, dim={model.hidden_dim}, layers={model.num_layers}")

    kb_mgr = KBManager(config)
    engine = QueryEngine(model, config)
    editor = KBEditor(model)
    router = KBRouter(model, kb_mgr, config)

    class System:
        pass

    s = System()
    s.config = config
    s.model = model
    s.model_name = model_name
    s.kb_mgr = kb_mgr
    s.engine = engine
    s.editor = editor
    s.router = router
    s.sim = cosine_sim

    yield s

    model.unload()


# ========================================================================
# 1. SEMANTIC DISCRIMINATION
#    The model must distinguish unrelated concepts
# ========================================================================

class TestSemanticDiscrimination:
    """Verify the model actually separates unrelated meanings."""

    def test_same_word_identical(self, system):
        """The same text must produce the exact same vector."""
        v1 = system.model.get_hidden("cat")
        v2 = system.model.get_hidden("cat")
        sim = system.sim(v1, v2)
        assert sim > 0.999, f"Same text should be identical, got {sim:.4f}"

    def test_similar_concepts_closer_than_unrelated(self, system):
        """'cat' should be closer to 'dog' than to 'quantum physics'."""
        v_cat = system.model.get_hidden("cat")
        v_dog = system.model.get_hidden("dog")
        v_quantum = system.model.get_hidden("quantum physics")

        sim_cat_dog = system.sim(v_cat, v_dog)
        sim_cat_quantum = system.sim(v_cat, v_quantum)

        print(f"  cat<->dog: {sim_cat_dog:.4f}, cat<->quantum: {sim_cat_quantum:.4f}")
        assert sim_cat_dog > sim_cat_quantum, \
            f"cat-dog ({sim_cat_dog:.4f}) should be closer than cat-quantum ({sim_cat_quantum:.4f})"

    @pytest.mark.xfail(
        reason="Requires strong multilingual model; Llama-3.2-3B base may not pass",
        strict=False,
    )
    def test_languages_share_meaning(self, system):
        """Cross-lingual: 'water' vs 'acqua' (Italian) vs 'fire'.

        Requires a model with genuine multilingual capacity.
        TinyLlama and Llama-3.2-3B-base are primarily English-trained
        and may not embed Italian text in the same semantic space.
        """
        v_water = system.model.get_hidden("water, a clear liquid for drinking")
        v_acqua = system.model.get_hidden("acqua, un liquido trasparente da bere")
        v_fire = system.model.get_hidden("fire, hot flames that burn and destroy")

        sim_water_acqua = system.sim(v_water, v_acqua)
        sim_water_fire = system.sim(v_water, v_fire)

        print(f"  water<->acqua: {sim_water_acqua:.4f}, water<->fire: {sim_water_fire:.4f}")
        assert sim_water_acqua > sim_water_fire, \
            f"water-acqua ({sim_water_acqua:.4f}) should be closer than water-fire ({sim_water_fire:.4f})"

    def test_numbers_form_cluster(self, system):
        """Numbers should be more similar to each other than to colors."""
        v_one = system.model.get_hidden("one")
        v_two = system.model.get_hidden("two")
        v_three = system.model.get_hidden("three")
        v_red = system.model.get_hidden("red")
        v_blue = system.model.get_hidden("blue")

        # Average intra-cluster similarity
        num_sims = [
            system.sim(v_one, v_two),
            system.sim(v_one, v_three),
            system.sim(v_two, v_three),
        ]
        # Average cross-cluster similarity
        cross_sims = [
            system.sim(v_one, v_red),
            system.sim(v_two, v_blue),
            system.sim(v_three, v_red),
        ]
        avg_num = sum(num_sims) / len(num_sims)
        avg_cross = sum(cross_sims) / len(cross_sims)

        print(f"  numbers intra-sim: {avg_num:.4f}, numbers<->colors: {avg_cross:.4f}")
        assert avg_num > avg_cross, \
            f"Numbers should cluster ({avg_num:.4f}) tighter than numbers-colors ({avg_cross:.4f})"


# ========================================================================
# 2. RELATIONAL STRUCTURE
#    Country-capital pairs should share geometric structure
# ========================================================================

class TestRelationalStructure:
    """Verify that semantic relations have consistent vector geometry."""

    @pytest.fixture()
    def geo_kb(self, system):
        from kb4096d.kb_store import KnowledgeBase
        kb = KnowledgeBase("geo_deep", hidden_dim=system.model.hidden_dim)
        # Use descriptive texts for indexing to produce richer vectors
        # that match well against sentence-level queries.
        concepts = {
            "France": "France, a country in Western Europe",
            "Paris": "Paris, the capital city of France",
            "Italy": "Italy, a country in Southern Europe",
            "Rome": "Rome, the capital city of Italy",
            "Japan": "Japan, an island nation in East Asia",
            "Tokyo": "Tokyo, the capital city of Japan",
            "Germany": "Germany, a country in Central Europe",
            "Berlin": "Berlin, the capital city of Germany",
            "Spain": "Spain, a country in Southern Europe",
            "Madrid": "Madrid, the capital city of Spain",
            "Brazil": "Brazil, a country in South America",
            "Brasilia": "Brasilia, the capital city of Brazil",
        }
        names = list(concepts.keys())
        texts = list(concepts.values())
        vecs = system.model.get_hidden_batch(texts)
        for name, vec in zip(names, vecs):
            kb.add_concept(name, vec)
        return kb

    def test_capital_closer_to_own_country(self, system, geo_kb):
        """Descriptive capital sentence should be closer to own country.

        Single-word tokens like 'Paris' and 'France' are too short
        for TinyLlama to reliably distinguish. Using richer descriptions
        provides the model with enough context.
        """
        v_paris = system.model.get_hidden("Paris, the capital city of France")
        v_france = system.model.get_hidden("France, a country in Western Europe")
        v_japan = system.model.get_hidden("Japan, an island nation in East Asia")

        sim_paris_france = system.sim(v_paris, v_france)
        sim_paris_japan = system.sim(v_paris, v_japan)

        print(f"  'Paris capital of France'<->France: {sim_paris_france:.4f}")
        print(f"  'Paris capital of France'<->Japan:  {sim_paris_japan:.4f}")
        assert sim_paris_france > sim_paris_japan, \
            f"Paris-France ({sim_paris_france:.4f}) should beat Paris-Japan ({sim_paris_japan:.4f})"

    def test_country_capital_offsets_parallel(self, system, geo_kb):
        """Test if capital offsets share geometric structure.

        The classic word2vec hypothesis: (Paris-France) ≈ (Rome-Italy).
        With descriptive inputs, the model should encode the capital-of
        relationship more consistently.
        """
        # Use descriptive phrases for richer representations
        pairs = [
            ("France is a country in Europe", "Paris is the capital of France"),
            ("Italy is a country in Europe", "Rome is the capital of Italy"),
            ("Japan is a country in Asia", "Tokyo is the capital of Japan"),
        ]
        offsets = []
        from kb4096d.utils import normalize_vector
        for country_text, capital_text in pairs:
            v_country = system.model.get_hidden(country_text)
            v_capital = system.model.get_hidden(capital_text)
            offsets.append(normalize_vector(v_capital - v_country))

        # Capital offsets should correlate with each other
        sim_01 = system.sim(offsets[0], offsets[1])
        sim_02 = system.sim(offsets[0], offsets[2])
        sim_12 = system.sim(offsets[1], offsets[2])

        # Noise: an unrelated direction
        v_cat = system.model.get_hidden("a cat is a small animal")
        v_math = system.model.get_hidden("mathematics is the study of numbers")
        noise_offset = normalize_vector(v_cat - v_math)
        sim_noise = system.sim(offsets[0], noise_offset)

        avg_capital = (sim_01 + sim_02 + sim_12) / 3

        print(f"  Capital offsets (descriptive):")
        print(f"    FR->PA vs IT->RO: {sim_01:.4f}")
        print(f"    FR->PA vs JP->TK: {sim_02:.4f}")
        print(f"    IT->RO vs JP->TK: {sim_12:.4f}")
        print(f"    Average: {avg_capital:.4f}")
        print(f"    vs noise (cat-math): {sim_noise:.4f}")

        # At least the average of capital offsets should be positive
        # (indicating some shared structure)
        assert avg_capital > 0.0, \
            f"Capital offsets should have positive average similarity ({avg_capital:.4f})"

    def test_search_finds_capital_for_country_query(self, system, geo_kb):
        """Searching 'capital of France' should rank Paris high."""
        results = system.engine.search("the capital city of France", geo_kb, top_k=6)
        names = [r[0] for r in results]
        scores = {r[0]: r[1] for r in results}

        print(f"  Search 'capital city of France':")
        for name, score in results:
            print(f"    {name}: {score:.4f}")

        # Paris should be in top 3
        assert "Paris" in names[:3], \
            f"Paris should be in top 3 for 'capital of France', got {names[:3]}"

    def test_search_finds_country_for_capital_query(self, system, geo_kb):
        """Searching 'Tokyo is the capital of which country' should rank Japan high."""
        results = system.engine.search("Tokyo is the capital of which country", geo_kb, top_k=6)
        names = [r[0] for r in results]

        print(f"  Search 'Tokyo is the capital of which country':")
        for name, score in results:
            print(f"    {name}: {score:.4f}")

        assert "Japan" in names[:3] or "Tokyo" in names[:2], \
            f"Japan or Tokyo should be top for this query, got {names[:3]}"


# ========================================================================
# 3. ANALOGY
#    Vector arithmetic should produce semantically meaningful results
# ========================================================================

class TestAnalogy:
    """Verify that analogy operations (a + (c - b)) work semantically."""

    @pytest.fixture()
    def geo_kb(self, system):
        from kb4096d.kb_store import KnowledgeBase
        kb = KnowledgeBase("analogy_test", hidden_dim=system.model.hidden_dim)
        concepts = [
            "France", "Paris", "Italy", "Rome", "Japan", "Tokyo",
            "Germany", "Berlin", "Spain", "Madrid", "Brazil", "Brasilia",
        ]
        vecs = system.model.get_hidden_batch(concepts)
        for name, vec in zip(concepts, vecs):
            kb.add_concept(name, vec)
        return kb

    def test_analogy_japan_capital(self, system, geo_kb):
        """Japan + (Paris - France) should be near Tokyo.

        This is the classic word2vec analogy test applied to our KB.
        """
        vec, neighbors = system.editor.analogy(
            geo_kb, "_japan_capital",
            base="Japan", source="France", target="Paris"
        )

        print(f"  Analogy: Japan + (Paris - France) =")
        for name, score in neighbors:
            if name != "_japan_capital":
                print(f"    {name}: {score:.4f}")

        # Get the non-self neighbors
        real_neighbors = [(n, s) for n, s in neighbors if n != "_japan_capital"]
        top_names = [n for n, s in real_neighbors[:3]]

        # Tokyo should be reasonably close. Even if not #1, it should be
        # closer than unrelated concepts
        sim_to_tokyo = system.sim(vec, geo_kb.concepts["Tokyo"].vector)
        sim_to_brazil = system.sim(vec, geo_kb.concepts["Brazil"].vector)

        print(f"  Result similarity to Tokyo: {sim_to_tokyo:.4f}")
        print(f"  Result similarity to Brazil: {sim_to_brazil:.4f}")

        assert sim_to_tokyo > sim_to_brazil, \
            f"Analogy result should be closer to Tokyo ({sim_to_tokyo:.4f}) " \
            f"than Brazil ({sim_to_brazil:.4f})"

    def test_analogy_germany_capital(self, system, geo_kb):
        """Germany + (Rome - Italy) should be near Berlin."""
        vec, neighbors = system.editor.analogy(
            geo_kb, "_de_capital",
            base="Germany", source="Italy", target="Rome"
        )

        real_neighbors = [(n, s) for n, s in neighbors if n != "_de_capital"]
        print(f"  Analogy: Germany + (Rome - Italy) =")
        for name, score in real_neighbors[:5]:
            print(f"    {name}: {score:.4f}")

        sim_to_berlin = system.sim(vec, geo_kb.concepts["Berlin"].vector)
        sim_to_tokyo = system.sim(vec, geo_kb.concepts["Tokyo"].vector)

        print(f"  Result -> Berlin: {sim_to_berlin:.4f}, -> Tokyo: {sim_to_tokyo:.4f}")

        # Berlin should be closer than a random other capital
        assert sim_to_berlin > sim_to_tokyo, \
            f"Should be closer to Berlin ({sim_to_berlin:.4f}) than Tokyo ({sim_to_tokyo:.4f})"


# ========================================================================
# 4. NOVEL CONCEPTS
#    Inject entirely new knowledge and verify it's retrievable
# ========================================================================

class TestNovelConcepts:
    """Inject invented concepts and verify the KB stores and retrieves them."""

    def test_invented_entity(self, system):
        """Create a KB with a fictional country, verify it's searchable."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("fiction", hidden_dim=system.model.hidden_dim)

        # Real concepts for context
        real = ["France", "Paris", "Germany", "Berlin", "Japan", "Tokyo"]
        vecs = system.model.get_hidden_batch(real)
        for name, vec in zip(real, vecs):
            kb.add_concept(name, vec)

        # Invented concept: "Zaltoria" described as a tropical island nation
        system.editor.add_concept(
            kb, "Zaltoria",
            text="Zaltoria is a small tropical island nation in the Pacific Ocean"
        )
        # Invented capital
        system.editor.add_concept(
            kb, "Zaltopolis",
            text="Zaltopolis is the capital city of the island nation Zaltoria"
        )

        # Search for the fictional country
        results = system.engine.search("tropical island nation", kb, top_k=8)
        names = [r[0] for r in results]

        print(f"  Search 'tropical island nation':")
        for name, score in results:
            print(f"    {name}: {score:.4f}")

        # Zaltoria should rank higher than European countries for this query
        zaltoria_rank = names.index("Zaltoria") if "Zaltoria" in names else len(names)
        france_rank = names.index("France") if "France" in names else len(names)

        assert zaltoria_rank < france_rank, \
            f"Zaltoria (rank {zaltoria_rank}) should rank above France (rank {france_rank}) " \
            f"for 'tropical island nation'"

    def test_inject_domain_knowledge(self, system):
        """Create a specialized KB about programming languages."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("programming", hidden_dim=system.model.hidden_dim)

        concepts = {
            "Python": "Python is a high-level interpreted programming language",
            "JavaScript": "JavaScript is a programming language used for web development",
            "Rust": "Rust is a systems programming language focused on memory safety",
            "SQL": "SQL is a language for managing relational databases",
            "HTML": "HTML is a markup language for creating web pages",
            "TensorFlow": "TensorFlow is a machine learning framework by Google",
            "React": "React is a JavaScript library for building user interfaces",
        }
        for name, desc in concepts.items():
            system.editor.add_concept(kb, name, text=desc)

        # Search for ML-related concept
        results = system.engine.search("machine learning framework", kb, top_k=7)
        names = [r[0] for r in results]

        print(f"  Search 'machine learning framework':")
        for name, score in results:
            print(f"    {name}: {score:.4f}")

        # TensorFlow and Python should rank higher than HTML
        tf_rank = names.index("TensorFlow") if "TensorFlow" in names else len(names)
        html_rank = names.index("HTML") if "HTML" in names else len(names)

        assert tf_rank < html_rank, \
            f"TensorFlow (rank {tf_rank}) should beat HTML (rank {html_rank}) " \
            f"for 'machine learning framework'"

        # Search for web development
        results2 = system.engine.search("web development frontend", kb, top_k=7)
        names2 = [r[0] for r in results2]

        print(f"  Search 'web development frontend':")
        for name, score in results2:
            print(f"    {name}: {score:.4f}")

        # JavaScript/React should rank higher than Rust for web queries
        js_rank = names2.index("JavaScript") if "JavaScript" in names2 else len(names2)
        rust_rank = names2.index("Rust") if "Rust" in names2 else len(names2)

        assert js_rank < rust_rank, \
            f"JavaScript (rank {js_rank}) should beat Rust (rank {rust_rank}) for web dev"


# ========================================================================
# 5. FACT CORRECTION
#    Modify a fact and verify the change is reflected
# ========================================================================

class TestFactCorrection:
    """Verify that editing a concept actually changes its semantic position."""

    def test_correct_fact_changes_similarity(self, system):
        """Move 'Paris' to be about 'a city in Japan' and verify it shifts."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("correction", hidden_dim=system.model.hidden_dim)
        concepts = ["France", "Paris", "Japan", "Tokyo"]
        vecs = system.model.get_hidden_batch(concepts)
        for name, vec in zip(concepts, vecs):
            kb.add_concept(name, vec)

        # Before correction
        sim_paris_france_before = system.sim(
            kb.concepts["Paris"].vector, kb.concepts["France"].vector
        )
        sim_paris_japan_before = system.sim(
            kb.concepts["Paris"].vector, kb.concepts["Japan"].vector
        )

        print(f"  BEFORE correction:")
        print(f"    Paris<->France: {sim_paris_france_before:.4f}")
        print(f"    Paris<->Japan:  {sim_paris_japan_before:.4f}")

        # Correct: redefine Paris as a Japanese concept
        system.editor.correct_fact(
            kb, "Paris",
            "Paris is a district in Tokyo, Japan, known for Japanese gardens"
        )

        # After correction
        sim_paris_france_after = system.sim(
            kb.concepts["Paris"].vector, kb.concepts["France"].vector
        )
        sim_paris_japan_after = system.sim(
            kb.concepts["Paris"].vector, kb.concepts["Japan"].vector
        )

        print(f"  AFTER correction (Paris = Japanese district):")
        print(f"    Paris<->France: {sim_paris_france_after:.4f}")
        print(f"    Paris<->Japan:  {sim_paris_japan_after:.4f}")

        # Paris should have moved closer to Japan and further from France
        shift_toward_japan = sim_paris_japan_after - sim_paris_japan_before
        shift_from_france = sim_paris_france_before - sim_paris_france_after

        print(f"  Shift toward Japan: {shift_toward_japan:+.4f}")
        print(f"  Shift from France:  {shift_from_france:+.4f}")

        assert shift_toward_japan > 0 or shift_from_france > 0, \
            "Correcting Paris to be Japanese should shift it toward Japan or away from France"

    def test_move_concept_shifts_position(self, system):
        """Use move_concept to push a concept in a new direction."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("move_test", hidden_dim=system.model.hidden_dim)
        for name in ["cat", "dog", "fish", "bird"]:
            system.editor.add_concept(kb, name)

        sim_cat_fish_before = system.sim(
            kb.concepts["cat"].vector, kb.concepts["fish"].vector
        )

        # Move cat toward "ocean marine underwater"
        system.editor.move_concept(kb, "cat", "ocean marine underwater fish", strength=0.5)

        sim_cat_fish_after = system.sim(
            kb.concepts["cat"].vector, kb.concepts["fish"].vector
        )

        print(f"  cat<->fish before move: {sim_cat_fish_before:.4f}")
        print(f"  cat<->fish after move:  {sim_cat_fish_after:.4f}")

        assert sim_cat_fish_after > sim_cat_fish_before, \
            f"Moving cat toward ocean should increase cat-fish similarity " \
            f"(before: {sim_cat_fish_before:.4f}, after: {sim_cat_fish_after:.4f})"


# ========================================================================
# 6. CROSS-DOMAIN SEPARATION & ROUTING
#    Different KBs should be distinguishable by topic
# ========================================================================

class TestCrossDomainRouting:
    """Verify that the router can distinguish between domain-specific KBs."""

    def test_router_selects_correct_kb(self, system):
        """Create geo and music KBs, verify router picks the right one."""
        from kb4096d.kb_store import KnowledgeBase

        # Geography KB
        geo_kb = KnowledgeBase("geography", hidden_dim=system.model.hidden_dim)
        geo_concepts = {
            "France": "France is a country in Western Europe",
            "Tokyo": "Tokyo is the capital of Japan",
            "Amazon": "The Amazon river flows through South America",
            "Alps": "The Alps are a mountain range in Europe",
        }
        for name, desc in geo_concepts.items():
            system.editor.add_concept(geo_kb, name, text=desc)
        system.kb_mgr.register(geo_kb)

        # Music KB
        music_kb = KnowledgeBase("music", hidden_dim=system.model.hidden_dim)
        music_concepts = {
            "guitar": "an electric guitar used in rock music",
            "symphony": "a symphony is an orchestral musical composition",
            "jazz": "jazz is a music genre that originated in New Orleans",
            "piano": "a piano is a keyboard musical instrument",
        }
        for name, desc in music_concepts.items():
            system.editor.add_concept(music_kb, name, text=desc)
        system.kb_mgr.register(music_kb)

        # Test routing
        geo_query = "What is the largest country in Europe?"
        music_query = "Who invented the electric guitar?"

        geo_results = system.router.route(geo_query, top_k=2)
        music_results = system.router.route(music_query, top_k=2)

        print(f"  Route '{geo_query[:40]}...':")
        for name, sim in geo_results:
            print(f"    {name}: {sim:.4f}")

        print(f"  Route '{music_query[:40]}...':")
        for name, sim in music_results:
            print(f"    {name}: {sim:.4f}")

        # Geography query should rank geo KB higher
        geo_names = [r[0] for r in geo_results]
        music_names = [r[0] for r in music_results]

        assert geo_names[0] == "geography", \
            f"Geography query should route to 'geography', got '{geo_names[0]}'"
        assert music_names[0] == "music", \
            f"Music query should route to 'music', got '{music_names[0]}'"

        # Cleanup
        system.kb_mgr.unload("geography")
        system.kb_mgr.unload("music")


# ========================================================================
# 7. INTERPOLATION SEMANTICS
#    Blending two concepts should land in between
# ========================================================================

class TestInterpolation:
    """Verify interpolation produces semantically intermediate results."""

    def test_interpolate_lands_between(self, system):
        """Interpolating 'hot' and 'cold' at 0.5 should be closer to both
        than either is to an unrelated concept like 'computer'."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("interp", hidden_dim=system.model.hidden_dim)
        for name in ["hot", "cold", "warm", "computer"]:
            system.editor.add_concept(kb, name)

        system.editor.interpolate(kb, "hot_cold_mix", "hot", "cold", alpha=0.5)

        mix = kb.concepts["hot_cold_mix"].vector
        v_hot = kb.concepts["hot"].vector
        v_cold = kb.concepts["cold"].vector
        v_warm = kb.concepts["warm"].vector
        v_computer = kb.concepts["computer"].vector

        sim_mix_hot = system.sim(mix, v_hot)
        sim_mix_cold = system.sim(mix, v_cold)
        sim_mix_warm = system.sim(mix, v_warm)
        sim_mix_computer = system.sim(mix, v_computer)

        print(f"  Interpolation hot<->cold at 0.5:")
        print(f"    mix<->hot:      {sim_mix_hot:.4f}")
        print(f"    mix<->cold:     {sim_mix_cold:.4f}")
        print(f"    mix<->warm:     {sim_mix_warm:.4f}")
        print(f"    mix<->computer: {sim_mix_computer:.4f}")

        # Mix should be closer to temperature words than to 'computer'
        assert sim_mix_hot > sim_mix_computer, \
            f"hot-cold mix should be closer to 'hot' ({sim_mix_hot:.4f}) " \
            f"than 'computer' ({sim_mix_computer:.4f})"
        assert sim_mix_cold > sim_mix_computer, \
            f"hot-cold mix should be closer to 'cold' ({sim_mix_cold:.4f}) " \
            f"than 'computer' ({sim_mix_computer:.4f})"


# ========================================================================
# 8. RELATION EXTRACTION & RETRIEVAL
#    Explicit relations should be retrievable
# ========================================================================

class TestRelationExtraction:
    """Test that explicitly created relations work for retrieval."""

    def test_explicit_relation_retrieval(self, system):
        """Create explicit capital-of relations and verify relate() finds them."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("rel_test", hidden_dim=system.model.hidden_dim)

        pairs = [
            ("France", "Paris"), ("Italy", "Rome"),
            ("Japan", "Tokyo"), ("Egypt", "Cairo"),
        ]
        all_names = []
        for country, capital in pairs:
            all_names.extend([country, capital])

        vecs = system.model.get_hidden_batch(all_names)
        for name, vec in zip(all_names, vecs):
            kb.add_concept(name, vec)

        # Create explicit relations
        for country, capital in pairs:
            system.editor.create_relation(kb, country, "capital_of", capital)

        # Test: relate France capital_of should return Paris
        results = system.engine.relate("France", "capital_of", kb)
        print(f"  France capital_of:")
        for name, score in results:
            print(f"    {name}: {score:.4f}")

        assert results[0][0] == "Paris", \
            f"France capital_of should return Paris first, got {results[0][0]}"

        # Test: relate Japan capital_of should return Tokyo
        results2 = system.engine.relate("Japan", "capital_of", kb)
        print(f"  Japan capital_of:")
        for name, score in results2:
            print(f"    {name}: {score:.4f}")

        assert results2[0][0] == "Tokyo", \
            f"Japan capital_of should return Tokyo first, got {results2[0][0]}"

    def test_chain_follows_relations(self, system):
        """Chain through two relation types: continent -> country -> capital."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("chain_test", hidden_dim=system.model.hidden_dim)

        entities = [
            "Europe", "France", "Paris", "Italy", "Rome",
            "Asia", "Japan", "Tokyo", "China", "Beijing",
        ]
        vecs = system.model.get_hidden_batch(entities)
        for name, vec in zip(entities, vecs):
            kb.add_concept(name, vec)

        # Add relations
        system.editor.create_relation(kb, "Europe", "contains", "France")
        system.editor.create_relation(kb, "Europe", "contains", "Italy")
        system.editor.create_relation(kb, "Asia", "contains", "Japan")
        system.editor.create_relation(kb, "Asia", "contains", "China")
        system.editor.create_relation(kb, "France", "capital_of", "Paris")
        system.editor.create_relation(kb, "Italy", "capital_of", "Rome")
        system.editor.create_relation(kb, "Japan", "capital_of", "Tokyo")
        system.editor.create_relation(kb, "China", "capital_of", "Beijing")

        # Chain: Europe -> contains -> capital_of
        chain_results = system.engine.chain("Europe", ["contains", "capital_of"], kb)

        print(f"  Chain: Europe -> contains -> capital_of:")
        for i, step in enumerate(chain_results):
            print(f"    Step {i+1}: {step[:5]}")

        assert len(chain_results) == 2, "Should have 2 steps"

        # Step 1 should include European countries
        step1_names = [r[0] for r in chain_results[0]]
        print(f"  Step 1 results: {step1_names}")

        # Step 2 should include capitals
        step2_names = [r[0] for r in chain_results[1]]
        print(f"  Step 2 results: {step2_names}")

        # At least one European country should be in step 1
        european = {"France", "Italy"}
        found_european = european.intersection(set(step1_names))
        assert len(found_european) > 0, \
            f"Step 1 should contain European countries, got {step1_names}"


# ========================================================================
# 9. SAVE/LOAD PRESERVES SEMANTICS
#    After save and reload, vectors should be identical
# ========================================================================

class TestPersistence:
    """Verify that save/load preserves all semantic information exactly."""

    def test_save_load_preserves_vectors(self, system, tmp_path):
        """Save a KB with concepts and relations, reload, verify identical."""
        from kb4096d.kb_store import KnowledgeBase

        kb = KnowledgeBase("persist_test", hidden_dim=system.model.hidden_dim,
                           model_name="test")

        concepts = ["alpha", "beta", "gamma"]
        vecs = system.model.get_hidden_batch(concepts)
        for name, vec in zip(concepts, vecs):
            kb.add_concept(name, vec)
        system.editor.create_relation(kb, "alpha", "related_to", "beta")

        path = tmp_path / "test_persist.pt"
        kb.save(path)
        loaded = KnowledgeBase.load(path)

        # Vectors must be bitwise identical
        for name in concepts:
            original = kb.concepts[name].vector
            reloaded = loaded.concepts[name].vector
            assert torch.equal(original, reloaded), \
                f"Vector for '{name}' changed after save/load!"

        # Relations must be preserved
        assert len(loaded.relations) == 1
        assert loaded.relations[0].source == "alpha"
        assert loaded.relations[0].target == "beta"

        # Similarity must be identical
        sim_before = system.sim(kb.concepts["alpha"].vector, kb.concepts["beta"].vector)
        sim_after = system.sim(loaded.concepts["alpha"].vector, loaded.concepts["beta"].vector)
        assert abs(sim_before - sim_after) < 1e-6, \
            f"Similarity changed: {sim_before:.6f} vs {sim_after:.6f}"

        print(f"  Vectors identical after save/load: OK")
        print(f"  Similarity preserved: {sim_before:.6f} == {sim_after:.6f}")
