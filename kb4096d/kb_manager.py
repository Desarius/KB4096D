"""Multi-KB management: register, load, unload, save, scan."""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import KBConfig
from .kb_store import KnowledgeBase
from .model_loader import ModelManagerProtocol
from .utils import cosine_sim


class KBManager:
    """Manages multiple knowledge bases in memory and on disk."""

    def __init__(self, config: KBConfig):
        self.config = config
        self._kbs: Dict[str, KnowledgeBase] = {}

    @property
    def loaded_kbs(self) -> Dict[str, KnowledgeBase]:
        return self._kbs

    def create(
        self,
        name: str,
        description: str = "",
        hidden_dim: Optional[int] = None,
        model_name: str = "",
    ) -> KnowledgeBase:
        """Create a new empty KB and register it."""
        kb = KnowledgeBase(
            name=name,
            description=description,
            hidden_dim=hidden_dim,
            model_name=model_name,
        )
        self._kbs[name] = kb
        return kb

    def register(self, kb: KnowledgeBase) -> None:
        """Register an existing KB in the manager."""
        self._kbs[kb.name] = kb

    def get(self, name: str) -> Optional[KnowledgeBase]:
        """Get a loaded KB by name."""
        return self._kbs.get(name)

    def unload(self, name: str) -> bool:
        """Unload a KB from memory. Returns True if found."""
        if name in self._kbs:
            del self._kbs[name]
            return True
        return False

    def save(self, name: str, path: Optional[Path] = None) -> Path:
        """Save a KB to disk. Returns the file path."""
        kb = self._kbs.get(name)
        if kb is None:
            raise KeyError(f"KB '{name}' not loaded.")
        if path is None:
            path = self.config.kb_dir / f"{name}.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kb.save(path)
        return path

    def save_all(self) -> List[Path]:
        """Save all loaded KBs."""
        paths = []
        for name in self._kbs:
            paths.append(self.save(name))
        return paths

    def load_from_file(self, path: Path) -> KnowledgeBase:
        """Load a KB from a .pt file and register it."""
        path = Path(path)
        kb = KnowledgeBase.load(path)
        self._kbs[kb.name] = kb
        return kb

    def scan_directory(self, directory: Optional[Path] = None) -> List[str]:
        """Scan a directory for .pt files and list KB names (without loading).

        Returns list of KB names found.
        """
        directory = Path(directory) if directory else self.config.kb_dir
        found = []
        if directory.exists():
            for pt_file in sorted(directory.glob("*.pt")):
                try:
                    data = torch.load(pt_file, map_location="cpu", weights_only=False)
                    name = data.get("name", pt_file.stem)
                    found.append(name)
                except Exception:
                    continue
        return found

    def load_all_from_directory(self, directory: Optional[Path] = None) -> List[str]:
        """Load all .pt KBs from directory. Returns names of loaded KBs."""
        directory = Path(directory) if directory else self.config.kb_dir
        loaded = []
        if directory.exists():
            for pt_file in sorted(directory.glob("*.pt")):
                try:
                    kb = KnowledgeBase.load(pt_file)
                    self._kbs[kb.name] = kb
                    loaded.append(kb.name)
                except Exception as e:
                    print(f"Warning: failed to load {pt_file}: {e}")
        return loaded

    def find_relevant(
        self,
        query_vector: torch.Tensor,
        top_k: int = 3,
    ) -> List[tuple]:
        """Find KBs most relevant to a query vector via centroid comparison.

        Returns list of (kb_name, similarity) sorted descending.
        """
        results = []
        for name, kb in self._kbs.items():
            centroid = kb.centroid
            if centroid is not None:
                sim = cosine_sim(query_vector, centroid)
                results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def list_kbs(self) -> List[Dict]:
        """List all loaded KBs with summary info."""
        info = []
        for name, kb in self._kbs.items():
            info.append({
                "name": name,
                "description": kb.description,
                "concepts": len(kb.concepts),
                "relations": len(kb.relations),
                "hidden_dim": kb.hidden_dim,
                "model": kb.model_name,
            })
        return info

    def __contains__(self, name: str) -> bool:
        return name in self._kbs

    def __len__(self) -> int:
        return len(self._kbs)
