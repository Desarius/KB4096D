"""Text generation guided by KB knowledge via model hooks.

The generator installs a forward hook on the extraction layer that
blends KB-derived vectors into the model's hidden states during generation.
Hooks are ALWAYS cleaned up with try/finally to prevent model corruption.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import KBConfig
from .kb_store import KnowledgeBase
from .model_loader import ModelManager
from .query_engine import QueryEngine
from .utils import cosine_sim_batch, normalize_vector


class KBGenerator:
    """Generate text with KB knowledge injected via hooks."""

    def __init__(
        self,
        model_mgr: ModelManager,
        query_engine: QueryEngine,
        config: KBConfig,
    ):
        self.model_mgr = model_mgr
        self.query_engine = query_engine
        self.config = config

    def generate(
        self,
        prompt: str,
        kb: KnowledgeBase,
        max_new_tokens: Optional[int] = None,
        kb_influence: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text guided by KB knowledge.

        1. Find relevant concepts in KB for the prompt
        2. Install a hook that blends KB vectors into hidden states
        3. Generate text normally
        4. Always remove the hook (try/finally)

        Args:
            prompt: Input text to continue.
            kb: Knowledge base to use for guidance.
            max_new_tokens: Max tokens to generate.
            kb_influence: Strength of KB guidance (0.0-1.0).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text (without the prompt).
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        kb_influence = kb_influence if kb_influence is not None else self.config.kb_influence
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        model = self.model_mgr.get_model()
        tokenizer = self.model_mgr.get_tokenizer()
        extraction_layer = self.model_mgr.extraction_layer

        # Find relevant KB concepts for the prompt
        relevant = self.query_engine.search(prompt, kb, top_k=10, threshold=0.1)

        if not relevant:
            # No relevant concepts, generate without KB influence
            return self._generate_plain(prompt, max_new_tokens, temperature, top_p)

        # Build the KB guidance vector: weighted average of relevant concept vectors
        guidance_vec = self._build_guidance_vector(kb, relevant)
        guidance_vec = guidance_vec.to(model.device)

        # Get the target layer module
        target_layer = self._get_layer_module(model, extraction_layer)
        if target_layer is None:
            print("Warning: Could not find target layer, generating without KB.")
            return self._generate_plain(prompt, max_new_tokens, temperature, top_p)

        # Install hook and generate with try/finally for cleanup
        hook_handle = None
        try:
            def kb_hook(module, input, output):
                """Blend KB guidance into hidden states."""
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Add KB guidance to the last token position
                # (the one being generated)
                guidance = guidance_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                guidance = guidance.expand(-1, hidden_states.size(1), -1)
                # Match dtype of hidden states (model may use float16/bfloat16)
                guidance = guidance.to(dtype=hidden_states.dtype)

                modified = hidden_states + kb_influence * guidance

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified

            hook_handle = target_layer.register_forward_hook(kb_hook)

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].size(1):]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

        finally:
            # ALWAYS remove the hook
            if hook_handle is not None:
                hook_handle.remove()

        return generated

    def _generate_plain(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate without KB influence."""
        model = self.model_mgr.get_model()
        tokenizer = self.model_mgr.get_tokenizer()

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _build_guidance_vector(
        self,
        kb: KnowledgeBase,
        relevant: List[Tuple[str, float]],
    ) -> torch.Tensor:
        """Build a guidance vector from relevant concepts, weighted by similarity."""
        vectors = []
        weights = []
        for name, sim in relevant:
            concept = kb.get_concept(name)
            if concept is not None:
                vectors.append(concept.vector)
                weights.append(sim)

        if not vectors:
            return torch.zeros(self.model_mgr.hidden_dim)

        # Weighted average
        stacked = torch.stack(vectors)  # (N, D)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        weight_tensor = weight_tensor / weight_tensor.sum()
        guidance = (stacked * weight_tensor.unsqueeze(1)).sum(dim=0)

        return normalize_vector(guidance)

    def _get_layer_module(self, model, layer_idx: int):
        """Get the module for a specific transformer layer.

        Supports:
        - Standard: model.model.layers (Llama, Mistral)
        - GPT-style: model.transformer.h
        - Multimodal: model.language_model.model.layers (Llama-4)
        """
        layers = None

        # Try multimodal (Llama-4): model.language_model.model.layers
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                layers = lm.model.layers

        # Try standard: model.model.layers
        if layers is None and hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "layers"):
                layers = inner.layers
            elif hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                layers = inner.decoder.layers

        # Try GPT-style: model.transformer.h
        if layers is None and hasattr(model, "transformer"):
            inner = model.transformer
            if hasattr(inner, "h"):
                layers = inner.h
            elif hasattr(inner, "layers"):
                layers = inner.layers

        if layers is None:
            return None

        # layer_idx may equal num_layers (the output of the last layer),
        # in which case we hook the last decoder layer
        idx = min(layer_idx, len(layers) - 1)
        return layers[idx]
