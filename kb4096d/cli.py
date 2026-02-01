"""Interactive REPL for the KB4096D system."""

import cmd
import shlex
from pathlib import Path
from typing import Optional

from .config import KBConfig
from .model_loader import ModelManager
from .kb_manager import KBManager
from .query_engine import QueryEngine
from .editor import KBEditor
from .router import KBRouter
from .generator import KBGenerator
from .extractor import extract_concepts, extract_relation, build_kb
from .utils import format_results


class KB4096DCLI(cmd.Cmd):
    """Interactive command-line interface for KB4096D."""

    intro = (
        "\n=== KB4096D: Modular AI Knowledge Base ===\n"
        "Type 'help' for commands. Start with 'init' to load a model.\n"
    )
    prompt = "kb4096d> "

    def __init__(self, config: KBConfig):
        super().__init__()
        self.config = config
        self.model_mgr = ModelManager(config)
        self.kb_manager = KBManager(config)
        self.query_engine: Optional[QueryEngine] = None
        self.editor: Optional[KBEditor] = None
        self.router: Optional[KBRouter] = None
        self.generator: Optional[KBGenerator] = None
        self._active_kb: Optional[str] = None

    def _require_model(self) -> bool:
        if not self.model_mgr.is_loaded:
            print("Error: Model not loaded. Run 'init' first.")
            return False
        return True

    def _require_kb(self, name: Optional[str] = None) -> Optional[str]:
        """Resolve KB name (from arg or active). Returns name or None."""
        kb_name = name or self._active_kb
        if kb_name is None:
            print("Error: No KB specified and no active KB. Use --kb <name> or kb_use <name>.")
            return None
        if kb_name not in self.kb_manager:
            print(f"Error: KB '{kb_name}' not loaded.")
            return None
        return kb_name

    def _parse_kb_flag(self, args: str) -> tuple:
        """Extract --kb flag from args. Returns (remaining_args, kb_name_or_None)."""
        parts = shlex.split(args)
        kb_name = None
        filtered = []
        i = 0
        while i < len(parts):
            if parts[i] == "--kb" and i + 1 < len(parts):
                kb_name = parts[i + 1]
                i += 2
            else:
                filtered.append(parts[i])
                i += 1
        return " ".join(filtered), kb_name

    # ---- Model commands ----

    def do_init(self, args: str):
        """Load the AI model. Usage: init [model_name]"""
        if args.strip():
            self.config.model_name = args.strip()
        try:
            self.model_mgr.load()
            self.query_engine = QueryEngine(self.model_mgr, self.config)
            self.editor = KBEditor(self.model_mgr)
            self.router = KBRouter(self.model_mgr, self.kb_manager, self.config)
            self.generator = KBGenerator(self.model_mgr, self.query_engine, self.config)
            print("System ready.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def do_model_info(self, args: str):
        """Show info about the loaded model."""
        if not self._require_model():
            return
        print(f"Model: {self.model_mgr.model_name}")
        print(f"Hidden dim: {self.model_mgr.hidden_dim}")
        print(f"Layers: {self.model_mgr.num_layers}")
        print(f"Extraction layer: {self.model_mgr.extraction_layer}")
        print(f"Device: {self.model_mgr.device}")

    def do_unload(self, args: str):
        """Unload the model to free memory."""
        self.model_mgr.unload()
        self.query_engine = None
        self.editor = None
        self.router = None
        self.generator = None

    # ---- KB management commands ----

    def do_kb_create(self, args: str):
        """Create a new empty KB. Usage: kb_create <name> [description]"""
        parts = shlex.split(args)
        if not parts:
            print("Usage: kb_create <name> [description]")
            return
        name = parts[0]
        desc = " ".join(parts[1:]) if len(parts) > 1 else ""
        hidden_dim = self.model_mgr.hidden_dim if self.model_mgr.is_loaded else None
        model_name = self.model_mgr.model_name if self.model_mgr.is_loaded else ""
        kb = self.kb_manager.create(name, desc, hidden_dim, model_name)
        self._active_kb = name
        print(f"Created KB '{name}' (active)")

    def do_kb_list(self, args: str):
        """List all loaded KBs."""
        kbs = self.kb_manager.list_kbs()
        if not kbs:
            print("No KBs loaded.")
            return
        for info in kbs:
            active = " *" if info["name"] == self._active_kb else ""
            print(f"  {info['name']}{active}: {info['concepts']} concepts, "
                  f"{info['relations']} relations, dim={info['hidden_dim']}")

    def do_kb_use(self, args: str):
        """Set active KB. Usage: kb_use <name>"""
        name = args.strip()
        if not name:
            print(f"Active KB: {self._active_kb or '(none)'}")
            return
        if name not in self.kb_manager:
            print(f"KB '{name}' not loaded.")
            return
        self._active_kb = name
        print(f"Active KB: {name}")

    def do_kb_info(self, args: str):
        """Show details of a KB. Usage: kb_info [name]"""
        name = self._require_kb(args.strip() or None)
        if not name:
            return
        kb = self.kb_manager.get(name)
        print(f"Name: {kb.name}")
        print(f"Description: {kb.description}")
        print(f"Hidden dim: {kb.hidden_dim}")
        print(f"Model: {kb.model_name}")
        print(f"Concepts ({len(kb.concepts)}):")
        for cname in sorted(kb.concepts.keys()):
            print(f"  - {cname}")
        print(f"Relations ({len(kb.relations)}):")
        for rel in kb.relations:
            print(f"  - {rel.source} --[{rel.relation_type}]--> {rel.target}")
        print(f"Overrides: {len(kb.overrides)}")
        print(f"Created: {kb.created_at}")
        print(f"Modified: {kb.modified_at}")

    def do_kb_save(self, args: str):
        """Save a KB to disk. Usage: kb_save [name] [path]"""
        parts = shlex.split(args)
        name = parts[0] if parts else self._active_kb
        name = self._require_kb(name)
        if not name:
            return
        path = Path(parts[1]) if len(parts) > 1 else None
        try:
            saved_path = self.kb_manager.save(name, path)
            print(f"Saved '{name}' to {saved_path}")
        except Exception as e:
            print(f"Error saving: {e}")

    def do_kb_load(self, args: str):
        """Load a KB from disk. Usage: kb_load <path_or_name>"""
        name = args.strip()
        if not name:
            print("Usage: kb_load <path_or_name>")
            return
        path = Path(name)
        if not path.suffix:
            path = self.config.kb_dir / f"{name}.pt"
        if not path.exists():
            print(f"File not found: {path}")
            return
        try:
            kb = self.kb_manager.load_from_file(path)
            self._active_kb = kb.name
            print(f"Loaded '{kb.name}' ({len(kb.concepts)} concepts, active)")
        except Exception as e:
            print(f"Error loading: {e}")

    def do_kb_scan(self, args: str):
        """Scan knowledge_bases/ for available .pt files."""
        names = self.kb_manager.scan_directory()
        if names:
            print("Available KBs on disk:")
            for n in names:
                loaded = " (loaded)" if n in self.kb_manager else ""
                print(f"  - {n}{loaded}")
        else:
            print("No .pt files found in knowledge_bases/")

    def do_kb_delete(self, args: str):
        """Unload a KB from memory. Usage: kb_delete <name>"""
        name = args.strip()
        if not name:
            print("Usage: kb_delete <name>")
            return
        if self.kb_manager.unload(name):
            if self._active_kb == name:
                self._active_kb = None
            print(f"Unloaded '{name}'")
        else:
            print(f"KB '{name}' not found.")

    # ---- Extraction commands ----

    def do_extract(self, args: str):
        """Extract concepts into a KB. Usage: extract <kb_name> concept1,concept2,..."""
        if not self._require_model():
            return
        parts = shlex.split(args)
        if len(parts) < 2:
            print("Usage: extract <kb_name> concept1,concept2,...")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        concepts = [c.strip() for c in parts[1].split(",") if c.strip()]
        if not concepts:
            print("No concepts provided.")
            return
        print(f"Extracting {len(concepts)} concepts...")
        self.editor.add_concepts_batch(kb, concepts)
        # Update hidden_dim if not set
        if kb.hidden_dim is None:
            kb.hidden_dim = self.model_mgr.hidden_dim
            kb.model_name = self.model_mgr.model_name
        print(f"Added {len(concepts)} concepts to '{name}':")
        for c in concepts:
            print(f"  + {c}")

    def do_extract_relations(self, args: str):
        """Extract relations. Usage: extract_relations <kb> src,rel,tgt [src,rel,tgt ...]"""
        if not self._require_model():
            return
        parts = shlex.split(args)
        if len(parts) < 2:
            print("Usage: extract_relations <kb> src,rel,tgt [src,rel,tgt ...]")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        count = 0
        for triple_str in parts[1:]:
            triple = [x.strip() for x in triple_str.split(",")]
            if len(triple) != 3:
                print(f"  Skipping invalid triple: {triple_str}")
                continue
            src, rel_type, tgt = triple
            rel_vec = extract_relation(self.model_mgr, src, rel_type, tgt)
            kb.add_relation(src, rel_type, tgt, rel_vec)
            print(f"  + {src} --[{rel_type}]--> {tgt}")
            count += 1
        print(f"Added {count} relations.")

    # ---- Query commands ----

    def do_search(self, args: str):
        """Search for concepts. Usage: search <query> [--kb name]"""
        if not self._require_model():
            return
        query, kb_flag = self._parse_kb_flag(args)
        if not query:
            print("Usage: search <query> [--kb name]")
            return
        name = self._require_kb(kb_flag)
        if not name:
            return
        kb = self.kb_manager.get(name)
        results = self.query_engine.search(query, kb)
        print(f"Search in '{name}' for '{query}':")
        print(format_results(results, prefix="  "))

    def do_relate(self, args: str):
        """Find related concepts. Usage: relate <concept> <relation> [--kb name]"""
        if not self._require_model():
            return
        text, kb_flag = self._parse_kb_flag(args)
        parts = shlex.split(text)
        if len(parts) < 2:
            print("Usage: relate <concept> <relation_type> [--kb name]")
            return
        concept = parts[0]
        rel_type = parts[1]
        name = self._require_kb(kb_flag)
        if not name:
            return
        kb = self.kb_manager.get(name)
        results = self.query_engine.relate(concept, rel_type, kb)
        print(f"'{concept}' {rel_type}:")
        print(format_results(results, prefix="  "))

    def do_chain(self, args: str):
        """Chain relations. Usage: chain <concept> rel1,rel2,... [--kb name]"""
        if not self._require_model():
            return
        text, kb_flag = self._parse_kb_flag(args)
        parts = shlex.split(text)
        if len(parts) < 2:
            print("Usage: chain <concept> rel1,rel2,... [--kb name]")
            return
        concept = parts[0]
        rels = [r.strip() for r in parts[1].split(",")]
        name = self._require_kb(kb_flag)
        if not name:
            return
        kb = self.kb_manager.get(name)
        chain_results = self.query_engine.chain(concept, rels, kb)
        for i, (rel, step) in enumerate(zip(rels, chain_results)):
            print(f"Step {i+1} ({rel}):")
            print(format_results(step, prefix="  "))

    # ---- Editing commands ----

    def do_add(self, args: str):
        """Add a concept. Usage: add <kb_name> <concept> [text_for_vector]"""
        if not self._require_model():
            return
        parts = shlex.split(args)
        if len(parts) < 2:
            print("Usage: add <kb_name> <concept> [text]")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        concept = parts[1]
        text = " ".join(parts[2:]) if len(parts) > 2 else None
        self.editor.add_concept(kb, concept, text)
        print(f"Added '{concept}' to '{name}'")

    def do_remove(self, args: str):
        """Remove a concept. Usage: remove <kb_name> <concept>"""
        parts = shlex.split(args)
        if len(parts) < 2:
            print("Usage: remove <kb_name> <concept>")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        if kb.remove_concept(parts[1]):
            print(f"Removed '{parts[1]}' from '{name}'")
        else:
            print(f"Concept '{parts[1]}' not found.")

    def do_move(self, args: str):
        """Move a concept towards a direction. Usage: move <kb> <concept> <direction> [strength]"""
        if not self._require_model():
            return
        parts = shlex.split(args)
        if len(parts) < 3:
            print("Usage: move <kb> <concept> <direction_text> [strength=0.3]")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        strength = float(parts[3]) if len(parts) > 3 else 0.3
        try:
            self.editor.move_concept(kb, parts[1], parts[2], strength)
            print(f"Moved '{parts[1]}' towards '{parts[2]}' (strength={strength})")
        except KeyError as e:
            print(f"Error: {e}")

    def do_interpolate(self, args: str):
        """Interpolate two concepts. Usage: interpolate <kb> <new_name> <concept_a> <concept_b> [alpha=0.5]"""
        if not self._require_model():
            return
        parts = shlex.split(args)
        if len(parts) < 4:
            print("Usage: interpolate <kb> <new_name> <concept_a> <concept_b> [alpha=0.5]")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        alpha = float(parts[4]) if len(parts) > 4 else 0.5
        try:
            self.editor.interpolate(kb, parts[1], parts[2], parts[3], alpha)
            print(f"Created '{parts[1]}' by interpolating {parts[2]} and {parts[3]} (alpha={alpha})")
        except KeyError as e:
            print(f"Error: {e}")

    def do_analogy(self, args: str):
        """Create a concept by analogy. Usage: analogy <kb> <new_name> base=X from=Y to=Z"""
        if not self._require_model():
            return
        parts = shlex.split(args)
        if len(parts) < 3:
            print("Usage: analogy <kb> <new_name> base=X from=Y to=Z")
            return
        name = self._require_kb(parts[0])
        if not name:
            return
        kb = self.kb_manager.get(name)
        new_name = parts[1]

        # Parse keyword arguments
        kwargs = {}
        for part in parts[2:]:
            if "=" in part:
                key, val = part.split("=", 1)
                kwargs[key] = val

        base = kwargs.get("base")
        src = kwargs.get("from")
        dst = kwargs.get("to")
        if not all([base, src, dst]):
            print("Error: need base=X from=Y to=Z")
            return

        try:
            vec, neighbors = self.editor.analogy(kb, new_name, base, src, dst)
            print(f"Created '{new_name}' by analogy: {base} + ({dst} - {src})")
            print("Nearest neighbors:")
            print(format_results(neighbors, prefix="  "))
        except KeyError as e:
            print(f"Error: {e}")

    def do_merge(self, args: str):
        """Merge KBs. Usage: merge <target_kb> <source_kb> [strategy=keep_target]"""
        parts = shlex.split(args)
        if len(parts) < 2:
            print("Usage: merge <target_kb> <source_kb> [keep_target|keep_source|average]")
            return
        target_name = self._require_kb(parts[0])
        source_name = self._require_kb(parts[1])
        if not target_name or not source_name:
            return
        strategy = parts[2] if len(parts) > 2 else "keep_target"
        target = self.kb_manager.get(target_name)
        source = self.kb_manager.get(source_name)
        changes = self.editor.merge_kbs(target, source, strategy)
        print(f"Merged '{source_name}' into '{target_name}': {changes} changes")

    # ---- Generation commands ----

    def do_generate(self, args: str):
        """Generate text guided by KB. Usage: generate <prompt> [--kb name]"""
        if not self._require_model():
            return
        prompt, kb_flag = self._parse_kb_flag(args)
        if not prompt:
            print("Usage: generate <prompt> [--kb name]")
            return
        name = self._require_kb(kb_flag)
        if not name:
            return
        kb = self.kb_manager.get(name)
        print(f"Generating with KB '{name}'...")
        try:
            result = self.generator.generate(prompt, kb)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {result}")
        except Exception as e:
            print(f"Error generating: {e}")

    # ---- Router commands ----

    def do_route(self, args: str):
        """Auto-detect best KB for a query. Usage: route <query>"""
        if not self._require_model():
            return
        if not args.strip():
            print("Usage: route <query>")
            return
        results = self.router.route(args.strip(), top_k=3)
        if results:
            print("Best KBs for query:")
            for name, sim in results:
                print(f"  {name}: {sim:.4f}")
        else:
            print("No relevant KBs found.")

    # ---- Utility commands ----

    def do_hidden(self, args: str):
        """Show hidden state info for text. Usage: hidden <text>"""
        if not self._require_model():
            return
        if not args.strip():
            print("Usage: hidden <text>")
            return
        vec = self.model_mgr.get_hidden(args.strip())
        print(f"Text: {args.strip()}")
        print(f"Vector shape: {vec.shape}")
        print(f"Norm: {vec.norm().item():.4f}")
        print(f"Mean: {vec.mean().item():.6f}")
        print(f"Std: {vec.std().item():.6f}")
        print(f"Min: {vec.min().item():.6f}, Max: {vec.max().item():.6f}")

    def do_similarity(self, args: str):
        """Cosine similarity between two texts. Usage: similarity <text1> | <text2>"""
        if not self._require_model():
            return
        if "|" not in args:
            print("Usage: similarity <text1> | <text2>")
            return
        parts = args.split("|", 1)
        t1 = parts[0].strip()
        t2 = parts[1].strip()
        from .utils import cosine_sim
        v1 = self.model_mgr.get_hidden(t1)
        v2 = self.model_mgr.get_hidden(t2)
        sim = cosine_sim(v1, v2)
        print(f"similarity('{t1}', '{t2}') = {sim:.4f}")

    def do_quit(self, args: str):
        """Exit the CLI."""
        print("Goodbye.")
        return True

    def do_exit(self, args: str):
        """Exit the CLI."""
        return self.do_quit(args)

    do_EOF = do_quit

    def default(self, line: str):
        print(f"Unknown command: {line}. Type 'help' for commands.")

    def emptyline(self):
        pass
