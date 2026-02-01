"""Configuration dataclass for the KB4096D system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KBConfig:
    """Global configuration for the KB4096D system."""

    # Model settings
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # Extraction settings
    extraction_layer: Optional[int] = None  # None = auto (last layer)
    pooling: str = "last_token"  # "last_token", "mean"
    batch_size: int = 16

    # KB storage
    kb_dir: Path = field(default_factory=lambda: Path("knowledge_bases"))

    # Search settings
    default_top_k: int = 5
    similarity_threshold: float = 0.3

    # Generation settings
    max_new_tokens: int = 128
    kb_influence: float = 0.7  # 0.0 = no KB, 1.0 = full KB override
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self):
        self.kb_dir = Path(self.kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def resolve_dtype(self):
        """Resolve dtype string to torch dtype."""
        import torch
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtypes.get(self.dtype, torch.float16)
