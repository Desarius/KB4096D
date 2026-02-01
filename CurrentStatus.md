# Test Suite Overview

This document describes the full test suite for the kb4096d project: what each test does, how it works, and what it validates.

## Architecture

The test suite is split into two tiers:

| Tier | Files | Data source | Requires GPU/model |
|------|-------|-------------|-------------------|
| **Unit tests** (mock) | `test_editor.py`, `test_kb_store.py`, `test_query_engine.py` | Synthetic vectors via `MockModelManager` | No |
| **Semantic tests** (real model) | `test_integration.py`, `test_semantic_deep.py` | Real hidden states from a transformer | Yes |

Unit tests always pass because they use deterministic hash-seeded vectors. Semantic tests validate that the model actually captures meaning; their pass/fail depends on model quality.

### Shared fixtures (`conftest.py`)

- **`MockModelManager`**: Replaces the real model for unit tests. Converts text to a deterministic vector by SHA-256 hashing the input and using the hash as a seed for `torch.randn`, then L2-normalizing. Same text always produces the same vector, different text produces unrelated vectors. Dimension: 256.
- **`mock_model`**: Pytest fixture returning a `MockModelManager` instance.
- **`config`**: Pytest fixture returning a `KBConfig` with a temporary `kb_dir`.
- **`populated_kb`**: Pytest fixture returning a `KnowledgeBase` pre-loaded with six geography concepts (Francia, Parigi, Italia, Roma, Giappone, Tokyo) using mock vectors.

---

## Tier 1: Unit Tests (mock data)

### `test_kb_store.py` -- KnowledgeBase data structure (12 tests)

Tests the core storage layer using hand-crafted `torch` tensors (dimension 4 or 256). No model involved.

| Test | What it validates |
|------|-------------------|
| `test_create_empty` | A new KB has the correct name, description, hidden_dim, and starts with zero concepts and relations. |
| `test_add_concept` | Adding a concept stores it with the correct vector shape. |
| `test_remove_concept` | Removing a concept deletes it; removing again returns `False`. |
| `test_remove_concept_cleans_relations` | When a concept is removed, any relation that references it (as source or target) is also deleted. |
| `test_add_relation` | A relation stores source, type, and target correctly. |
| `test_get_concept_vectors` | `get_concept_vectors()` returns a stacked tensor with shape `(n_concepts, dim)`. |
| `test_get_concept_vectors_empty` | On an empty KB, `get_concept_vectors()` returns shape `(0, dim)`. |
| `test_centroid` | The centroid of two orthogonal unit vectors is their element-wise mean. |
| `test_centroid_empty` | Centroid of an empty KB is `None`. |
| `test_save_load` | Save a KB with concepts, relations, and overrides to disk; reload and verify all fields (name, description, hidden_dim, model_name, vectors, relation types, override layer/strength) are identical. |
| `test_get_relations_for` | `get_relations_for("b")` returns all relations where "b" appears as source or target. |
| `test_get_relations_by_type` | `get_relations_by_type("likes")` filters relations by their type string. |
| `test_repr` | The `__repr__` string contains the KB name and dimension. |

### `test_editor.py` -- KBEditor operations (17 tests)

Tests every editor operation using `MockModelManager` and `populated_kb`. Vectors are deterministic but semantically meaningless, so these tests verify API contracts and data mutations, not semantic correctness.

| Test | What it validates |
|------|-------------------|
| `test_add_concept` | `add_concept("Brasile")` inserts a new entry into the KB using the model to generate its vector. |
| `test_add_concept_with_custom_text` | `add_concept("BRA", text="Il Brasile, paese del Sudamerica")` uses the custom description instead of the concept name to generate the vector. |
| `test_add_concepts_batch` | `add_concepts_batch` inserts multiple concepts in one call. |
| `test_remove_concept` | `remove_concept` deletes a concept and returns `True`; returns `False` for missing concepts. |
| `test_move_concept` | `move_concept("Francia", "Europa", strength=0.5)` shifts the vector so it is no longer identical to the original (cosine similarity < 0.99). |
| `test_move_concept_not_found` | Moving a non-existent concept raises `KeyError`. |
| `test_interpolate` | `interpolate("FrancItalia", "Francia", "Italia", alpha=0.5)` creates a new concept whose vector has positive similarity to both parents. |
| `test_interpolate_not_found` | Interpolating with a missing concept raises `KeyError`. |
| `test_analogy` | `analogy(kb, "capitale_giappone", base="Giappone", source="Francia", target="Parigi")` creates a new concept and returns a non-empty neighbor list. |
| `test_analogy_not_found` | Analogy with a missing concept raises `KeyError`. |
| `test_create_relation` | `create_relation("Francia", "capitale_di", "Parigi")` adds a typed relation retrievable by `get_relations_by_type`. |
| `test_create_relation_not_found` | Creating a relation with a missing concept raises `KeyError`. |
| `test_correct_fact` | `correct_fact("Parigi", "Paris, capital of France")` replaces the concept's vector so it differs from the original (cosine similarity < 0.99). |
| `test_correct_fact_not_found` | Correcting a missing concept raises `KeyError`. |
| `test_merge_kbs` | Merging two KBs with `keep_target` strategy: new concepts are copied, shared concepts keep the target's vector. Returns the count of new concepts added. |
| `test_merge_kbs_keep_source` | Merging with `keep_source` strategy overwrites the target's vector for shared concepts. |
| `test_merge_kbs_average` | Merging with `average` strategy produces a vector that is a normalized blend of both. |

### `test_query_engine.py` -- QueryEngine search and retrieval (10 tests)

Tests the query engine using `MockModelManager`. Since mock vectors are random, thresholds are set to 0.0 to avoid filtering out all results.

| Test | What it validates |
|------|-------------------|
| `test_search_returns_results` | `search("capitale del Giappone", kb)` returns a non-empty list of `(name, score)` tuples. |
| `test_search_empty_kb` | Searching an empty KB returns `[]`. |
| `test_search_by_vector` | `search_by_vector` with the exact vector of "Tokyo" returns "Tokyo" as the top result with similarity > 0.99. |
| `test_search_respects_top_k` | Setting `top_k=2` returns at most 2 results. |
| `test_search_respects_threshold` | With `threshold=0.99`, every returned result has similarity >= 0.99. |
| `test_relate_returns_results` | `relate("Giappone", "capitale_di", kb)` returns a non-empty list. |
| `test_relate_excludes_self` | The source concept ("Giappone") is excluded from `relate()` results. |
| `test_relate_with_explicit_relation` | When an explicit relation `(Giappone, capitale_di, Tokyo)` exists, `relate()` returns "Tokyo" as the first result. |
| `test_chain` | `chain("Italia", ["contiene", "capitale"], kb)` returns a list with 2 steps (one per relation type), each step being a list of results. |
| `test_multi_kb_search` | `multi_kb_search("test", [kb1, kb2])` searches across multiple KBs and returns results with a `kb_name` field (3-tuples). |

---

## Tier 2: Semantic Tests (real model)

These tests load a real transformer model and validate that hidden-state vectors carry actual semantic information. They are marked `@pytest.mark.slow` and require a GPU.

**Model selection**: Set the `KB_TEST_MODEL` environment variable to override the default (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`). Example:

```bash
KB_TEST_MODEL="meta-llama/Llama-3.2-1B" pytest tests/ -v
```

### `test_integration.py` -- Basic model integration (4 tests)

Validates that the model loads correctly and produces meaningful vectors.

| Test | What it validates |
|------|-------------------|
| `test_hidden_state_shape` | `get_hidden("test")` returns a 1-D tensor with shape `(hidden_dim,)` on CPU. |
| `test_hidden_state_batch` | `get_hidden_batch(["hello", "world"])` returns shape `(2, hidden_dim)`. |
| `test_similar_concepts_closer` | `cosine_sim(cat, dog) > cosine_sim(cat, "Python programming language")`. The model must place semantically related words closer together than unrelated ones. |
| `test_full_pipeline` | End-to-end: build a geography KB with 6 concepts and 1 relation, save to disk, reload, search for "capitale del Giappone", and verify results are non-empty. |

### `test_semantic_deep.py` -- Deep semantic validation (19 tests)

The most demanding tests. Each category probes a specific aspect of semantic knowledge representation.

#### 1. Semantic Discrimination (`TestSemanticDiscrimination`, 4 tests)

Can the model tell apart unrelated concepts?

| Test | What it validates |
|------|-------------------|
| `test_same_word_identical` | Encoding "cat" twice produces cosine similarity > 0.999 (determinism check). |
| `test_similar_concepts_closer_than_unrelated` | `sim(cat, dog) > sim(cat, "quantum physics")`. Basic semantic distance sanity. |
| `test_languages_share_meaning` *(xfail)* | `sim("water", "acqua") > sim("water", "fire")`. Cross-lingual semantic alignment. Marked `xfail` because most base models lack strong multilingual capacity. |
| `test_numbers_form_cluster` | Average pairwise similarity within {one, two, three} is higher than average similarity between numbers and colors {red, blue}. Tests categorical clustering. |

#### 2. Relational Structure (`TestRelationalStructure`, 4 tests)

Do country-capital pairs share consistent geometric structure?

| Test | What it validates |
|------|-------------------|
| `test_capital_closer_to_own_country` | Using descriptive sentences, "Paris, the capital city of France" is closer to "France, a country in Western Europe" than to "Japan, an island nation in East Asia". |
| `test_country_capital_offsets_parallel` | Computes the offset vector `capital - country` for France/Paris, Italy/Rome, Japan/Tokyo. The average pairwise cosine similarity between these offsets must be positive, indicating a shared "capital-of" direction. A noise offset (cat - math) is used as baseline. |
| `test_search_finds_capital_for_country_query` | Searching "the capital city of France" in a 12-concept geography KB returns "Paris" in the top 3. |
| `test_search_finds_country_for_capital_query` | Searching "Tokyo is the capital of which country" returns "Japan" in the top 3 or "Tokyo" in the top 2. |

#### 3. Analogy (`TestAnalogy`, 2 tests)

Does vector arithmetic produce semantically correct results?

| Test | What it validates |
|------|-------------------|
| `test_analogy_japan_capital` | Computes `Japan + (Paris - France)` and checks that the result is closer to Tokyo than to Brazil. Classic word2vec-style analogy. |
| `test_analogy_germany_capital` | Computes `Germany + (Rome - Italy)` and checks the result is closer to Berlin than to Tokyo. |

#### 4. Novel Concepts (`TestNovelConcepts`, 2 tests)

Can we inject entirely new knowledge and retrieve it?

| Test | What it validates |
|------|-------------------|
| `test_invented_entity` | Creates a fictional country "Zaltoria" (described as "a small tropical island nation in the Pacific Ocean") and its capital "Zaltopolis". Searching "tropical island nation" must rank Zaltoria above France. Proves the system can absorb and retrieve invented knowledge. |
| `test_inject_domain_knowledge` | Creates a programming languages KB with 7 concepts (Python, JavaScript, Rust, SQL, HTML, TensorFlow, React). Searching "machine learning framework" must rank TensorFlow above HTML. Searching "web development frontend" must rank JavaScript above Rust. Tests domain-specific retrieval accuracy. **Note**: This test fails on Llama-3.2-1B because the 1B model does not discriminate strongly enough between TensorFlow and general web technologies. |

#### 5. Fact Correction (`TestFactCorrection`, 2 tests)

Can we overwrite a fact and see the change reflected in the vector space?

| Test | What it validates |
|------|-------------------|
| `test_correct_fact_changes_similarity` | Redefines "Paris" as "a district in Tokyo, Japan, known for Japanese gardens". After correction, Paris must have shifted toward Japan or away from France (or both). |
| `test_move_concept_shifts_position` | Moves "cat" toward "ocean marine underwater fish" with strength 0.5. After the move, `sim(cat, fish)` must be higher than before. |

#### 6. Cross-Domain Routing (`TestCrossDomainRouting`, 1 test)

Are unrelated domains far apart, and can the router distinguish them?

| Test | What it validates |
|------|-------------------|
| `test_router_selects_correct_kb` | Creates a geography KB and a music KB. Routes "What is the largest country in Europe?" and verifies it selects the geography KB. Routes "Who invented the electric guitar?" and verifies it selects the music KB. |

#### 7. Interpolation (`TestInterpolation`, 1 test)

Does blending two concepts land semantically in between?

| Test | What it validates |
|------|-------------------|
| `test_interpolate_lands_between` | Interpolates "hot" and "cold" at alpha=0.5. The resulting blend must be closer to both "hot" and "cold" than to "computer". |

#### 8. Relation Extraction & Retrieval (`TestRelationExtraction`, 2 tests)

Do explicitly created relations enable correct retrieval?

| Test | What it validates |
|------|-------------------|
| `test_explicit_relation_retrieval` | Creates `capital_of` relations for 4 country-capital pairs. `relate("France", "capital_of", kb)` must return "Paris" first. `relate("Japan", "capital_of", kb)` must return "Tokyo" first. |
| `test_chain_follows_relations` | Builds a 2-hop relation graph: continent --contains--> country --capital_of--> capital. `chain("Europe", ["contains", "capital_of"], kb)` must return 2 steps, with step 1 containing at least one European country (France or Italy). |

#### 9. Persistence (`TestPersistence`, 1 test)

Does save/load preserve all semantic information?

| Test | What it validates |
|------|-------------------|
| `test_save_load_preserves_vectors` | Saves a KB with 3 concepts and 1 relation, reloads it, and checks: (1) vectors are bitwise identical (`torch.equal`), (2) relations are preserved, (3) cosine similarity between concepts is identical before and after (difference < 1e-6). |

---

## Test Results by Model

| Model | Unit tests (40) | Integration (4) | Semantic deep (19) | Total |
|-------|:---:|:---:|:---:|:---:|
| *Mock (no model)* | 40/40 | N/A | N/A | 40/40 |
| TinyLlama 1.1B | 40/40 | 0/4 (error) | 0/18+1xf (error) | 40/63 |
| Llama-3.2-1B | 40/40 | 4/4 | 17/18+1xf | 62/63 |
| Llama-3-8B | *not yet tested (pending access)* | | | |

**Known failure on Llama-3.2-1B**: `test_inject_domain_knowledge` -- the 1B model's hidden states do not separate "TensorFlow" (ML framework) strongly enough from general programming languages like HTML. The 8B model is expected to pass this test due to richer internal representations.

---

## Running the Tests

```bash
# All tests (requires GPU + model download)
pytest tests/ -v

# Unit tests only (no GPU needed)
pytest tests/ -v -m "not slow"

# Semantic tests only
pytest tests/ -v -m slow

# Override model
KB_TEST_MODEL="meta-llama/Meta-Llama-3-8B" pytest tests/ -v

# Skip multilingual test (xfail by default, but to exclude entirely)
pytest tests/ -v -k "not languages_share_meaning"
```

  
  
  ┌────────────────────────────────────┬───────────────────┬──────────────────────┐
  │              Metrics               │ TinyLlama (2048D) │ Llama-3.2-3B (3072D) │
  ├────────────────────────────────────┼───────────────────┼──────────────────────┤
  │ cat<->dog                          │ 0.48              │ 0.66                 │
  ├────────────────────────────────────┼───────────────────┼──────────────────────┤
  │ cat<->quantum                      │ 0.30              │ 0.48                 │
  ├────────────────────────────────────┼───────────────────┼──────────────────────┤
  │ Numbers cluster                    │ 0.70              │ 0.83                 │
  ├────────────────────────────────────┼───────────────────┼──────────────────────┤
  │ France-Italy                       │ 0.65              │ 0.93                 │
  ├────────────────────────────────────┼───────────────────┼──────────────────────┤
  │ Analogy Japan+(Paris-France)→Tokyo │ funziona          │ funziona             │
  ├────────────────────────────────────┼───────────────────┼──────────────────────┤
  │ Multilingual (acqua=water)         │ no                │ no (xfail)           │
  └────────────────────────────────────┴───────────────────┴──────────────────────┘
