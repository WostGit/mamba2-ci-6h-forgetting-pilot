"""Model loading and parameter selection for CPU-safe fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any, str]:
    """Load official checkpoint, with fallback model id for robustness."""
    model_name = config["model_name"]
    fallback = config.get("model_fallback_name")

    load_error = None
    for candidate in [model_name, fallback]:
        if not candidate:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(candidate)
            model.to(torch.device("cpu"))
            model.eval()
            return model, tokenizer, candidate
        except Exception as exc:  # noqa: BLE001
            load_error = exc

    raise RuntimeError(f"Failed to load model {model_name!r} and fallback {fallback!r}: {load_error}")


def apply_trainable_mode(model: Any, mode: str) -> int:
    """Freeze most parameters and return trainable count."""
    for p in model.parameters():
        p.requires_grad = False

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "lm_head_only":
        for name, p in model.named_parameters():
            if "lm_head" in name:
                p.requires_grad = True
    elif mode == "lm_head_plus_norm":
        for name, p in model.named_parameters():
            if "lm_head" in name or "norm" in name.lower():
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown trainable_mode={mode}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_adapter_state(model: Any, path: str | Path) -> None:
    """Save only trainable params to keep artifacts small."""
    trainable = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(trainable, Path(path))


def load_adapter_state(model: Any, path: str | Path) -> None:
    """Load trainable params from disk."""
    state = torch.load(Path(path), map_location="cpu")
    missing = []
    for name, p in model.named_parameters():
        if name in state:
            p.data.copy_(state[name])
        elif p.requires_grad:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing trainable tensors in adapter checkpoint: {missing[:5]}")
