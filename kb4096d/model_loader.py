"""Model loading and hidden state extraction."""

from typing import Optional, Protocol

import torch

from .config import KBConfig


class ModelManagerProtocol(Protocol):
    """Protocol for model manager (allows mocking in tests)."""

    @property
    def hidden_dim(self) -> int: ...

    @property
    def num_layers(self) -> int: ...

    @property
    def extraction_layer(self) -> int: ...

    @property
    def is_loaded(self) -> bool: ...

    def get_hidden(self, text: str) -> torch.Tensor: ...

    def get_hidden_batch(self, texts: list[str]) -> torch.Tensor: ...


class ModelManager:
    """Manages loading a transformer model and extracting hidden states."""

    def __init__(self, config: KBConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._hidden_dim: Optional[int] = None
        self._num_layers: Optional[int] = None
        self._extraction_layer: Optional[int] = None
        self._device: Optional[str] = None
        self._is_multimodal: bool = False

    @property
    def hidden_dim(self) -> int:
        assert self._hidden_dim is not None, "Model not loaded. Call load() first."
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        assert self._num_layers is not None, "Model not loaded. Call load() first."
        return self._num_layers

    @property
    def extraction_layer(self) -> int:
        assert self._extraction_layer is not None, "Model not loaded. Call load() first."
        return self._extraction_layer

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self.config.resolve_device()
        return self._device

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def load(self) -> None:
        """Load model and tokenizer. Reads hidden_dim from config, never hardcoded.

        Supports:
        - Standard causal LMs (Llama, TinyLlama, etc.)
        - Multimodal models with text sub-model (Llama-4 Scout/Maverick)
        - Quantized models (INT4/FP8 via compressed-tensors)
        - GPU/CPU offloading via device_map="auto"
        """
        if self._model is not None:
            print(f"Model already loaded: {self.config.model_name}")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        print(f"Loading model: {self.config.model_name}...")
        print(f"Device: {self.device}, dtype: {self.config.dtype}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left-padding is required for correct batch extraction with
        # autoregressive models. Right-padding corrupts hidden states
        # of shorter sequences in a batch (all collapse to same vector).
        self._tokenizer.padding_side = "left"

        # Check if this is a multimodal model (e.g. Llama-4)
        model_config = AutoConfig.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self._is_multimodal = hasattr(model_config, "text_config")

        # Llama-4 bug workaround: Llama4Config lacks pad_token_id at
        # the top level, but the model constructor expects it there.
        # Propagate from text_config if available.
        if self._is_multimodal:
            text_cfg = model_config.text_config
            for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
                if not hasattr(model_config, attr) or getattr(model_config, attr, None) is None:
                    val = getattr(text_cfg, attr, None)
                    if val is not None:
                        setattr(model_config, attr, val)

        load_kwargs = {
            "device_map": self.device if self.device != "cpu" else None,
            "trust_remote_code": True,
        }
        # transformers 5.x renamed torch_dtype -> dtype
        import transformers
        if hasattr(transformers, '__version__') and int(transformers.__version__.split('.')[0]) >= 5:
            load_kwargs["dtype"] = self.config.resolve_dtype()
        else:
            load_kwargs["torch_dtype"] = self.config.resolve_dtype()

        if self._is_multimodal:
            # Multimodal models (Llama-4) use ForConditionalGeneration,
            # not ForCausalLM. Load via AutoModel.
            from transformers import AutoModel
            print(f"Detected multimodal model (text_config present), using AutoModel")
            # Don't override dtype for quantized models (they have their own format)
            if model_config.quantization_config if hasattr(model_config, "quantization_config") else None:
                load_kwargs.pop("dtype", None)
                load_kwargs.pop("torch_dtype", None)
                print(f"Quantized model detected, using native quantization format")
            self._model = AutoModel.from_pretrained(
                self.config.model_name, config=model_config, **load_kwargs
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **load_kwargs
            )

        if self.device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()

        # Read dimensions from model config, never hardcode.
        # Multimodal models store text config in text_config sub-object.
        if self._is_multimodal:
            text_cfg = model_config.text_config
            self._hidden_dim = text_cfg.hidden_size
            self._num_layers = text_cfg.num_hidden_layers
        else:
            self._hidden_dim = self._model.config.hidden_size
            self._num_layers = self._model.config.num_hidden_layers

        # Auto extraction layer: last layer (best semantic discrimination)
        if self.config.extraction_layer is not None:
            self._extraction_layer = self.config.extraction_layer
        else:
            self._extraction_layer = self._num_layers  # last hidden_states index

        print(f"Model loaded: hidden_dim={self._hidden_dim}, "
              f"layers={self._num_layers}, "
              f"extraction_layer={self._extraction_layer}, "
              f"pooling={self.config.pooling}"
              f"{', multimodal=True' if self._is_multimodal else ''}")

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._hidden_dim = None
            self._num_layers = None
            self._extraction_layer = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded.")

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states according to configured strategy.

        Works correctly with both left-padding and right-padding.

        Args:
            hidden: (batch, seq_len, D)
            attention_mask: (batch, seq_len)

        Returns:
            (batch, D) pooled vectors.
        """
        if self.config.pooling == "last_token":
            # Find the rightmost non-padding position for each sequence.
            # Works for both left-padding ([PAD,PAD,BOS,tok]) and
            # right-padding ([BOS,tok,PAD,PAD]).
            batch_size, seq_len = attention_mask.shape
            positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  # (1, S)
            # Multiply positions by mask: padding positions become 0,
            # real positions keep their index. argmax finds the rightmost
            # non-zero position (= last real token).
            last_pos = (positions * attention_mask).argmax(dim=1)  # (batch,)
            batch_idx = torch.arange(batch_size, device=hidden.device)
            pooled = hidden[batch_idx, last_pos]  # (batch, D)
        else:
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, D)
        return pooled

    def _get_input_device(self) -> torch.device:
        """Get the device where input tensors should be sent.

        With device_map='auto', different layers may be on different devices.
        We need the device of the embedding layer (where inputs enter the model).
        """
        # Try hf_device_map first (set by device_map="auto")
        if hasattr(self._model, "hf_device_map"):
            # Find the embedding layer's device
            for key, dev in self._model.hf_device_map.items():
                if "embed" in key:
                    return torch.device(dev)
            # Fall back to first entry
            first_dev = next(iter(self._model.hf_device_map.values()))
            return torch.device(first_dev)
        return self._model.device

    def _forward_text(self, inputs: dict) -> object:
        """Run forward pass, handling both standard and multimodal models.

        For multimodal models, calls the language_model directly to get
        text hidden states without requiring image inputs.
        """
        if self._is_multimodal and hasattr(self._model, "language_model"):
            # Call the language model directly (skip vision encoder)
            lm = self._model.language_model
            return lm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        else:
            return self._model(
                **inputs,
                output_hidden_states=True,
            )

    @torch.no_grad()
    def get_hidden(self, text: str) -> torch.Tensor:
        """Extract hidden state for a single text. Returns CPU tensor of shape (D,)."""
        assert self._model is not None, "Model not loaded. Call load() first."

        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        outputs = self._forward_text(inputs)

        hidden = outputs.hidden_states[self._extraction_layer]  # (1, seq_len, D)
        pooled = self._pool_hidden(hidden, inputs["attention_mask"].to(hidden.device))  # (1, D)

        return pooled.squeeze(0).float().cpu()  # (D,)

    @torch.no_grad()
    def get_hidden_batch(self, texts: list[str]) -> torch.Tensor:
        """Extract hidden states for multiple texts. Returns CPU tensor of shape (N, D)."""
        assert self._model is not None, "Model not loaded. Call load() first."

        all_hidden = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_device = self._get_input_device()
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            outputs = self._forward_text(inputs)

            hidden = outputs.hidden_states[self._extraction_layer]
            pooled = self._pool_hidden(hidden, inputs["attention_mask"].to(hidden.device))

            all_hidden.append(pooled.float().cpu())

        return torch.cat(all_hidden, dim=0)  # (N, D)

    def get_model(self):
        """Access the underlying model (for generator hooks)."""
        assert self._model is not None, "Model not loaded."
        return self._model

    def get_tokenizer(self):
        """Access the tokenizer."""
        assert self._tokenizer is not None, "Model not loaded."
        return self._tokenizer
