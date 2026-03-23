"""Model loading and lightweight checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTNeoXTokenizerFast


class CheckpointPairingError(RuntimeError):
    """Raised when the configured checkpoint/model pairing is not trustworthy for CI."""


def load_model_and_tokenizer(cfg: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Mamba-2 model and tokenizer in CPU mode.

    Runtime-saving and honesty choice: fail fast when the advertised checkpoint
    metadata clearly disagrees with the loaded model shape, instead of burning
    runner time on a long CI job that cannot produce trustworthy results.
    """
    name = cfg["model"]["name"]
    tokenizer_name = cfg.get("model", {}).get("tokenizer_name", "EleutherAI/gpt-neox-20b")
    tok = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_cfg = AutoConfig.from_pretrained(name)
    hidden_size = getattr(model_cfg, 'hidden_size', None)
    intermediate_size = getattr(model_cfg, 'intermediate_size', None)
    if hidden_size is not None and intermediate_size is not None:
        ratio = intermediate_size / max(hidden_size, 1)
        if ratio >= 4.0:
            raise CheckpointPairingError(
                f"Configured checkpoint {name!r} resolves to hidden_size={hidden_size}, "
                f"intermediate_size={intermediate_size} (ratio={ratio:.2f}), which is inconsistent with the "
                "expected Mamba-2 130M scale. Refusing to run expensive CI on an unverified checkpoint/model pairing."
            )

    model = AutoModelForCausalLM.from_pretrained(name)
    model.to("cpu")
    model.train()
    return model, tok


def apply_lm_head_only_training(model: AutoModelForCausalLM) -> int:
    """Freeze all params except LM head (parameter-efficient CPU fallback)."""
    for p in model.parameters():
        p.requires_grad = False
    n_trainable = 0
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
            n_trainable += p.numel()
    return n_trainable


def save_trainable_checkpoint(model: AutoModelForCausalLM, path: str) -> None:
    """Save only trainable parameters for small artifacts."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    payload = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k in trainable_names
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_trainable_checkpoint(model: AutoModelForCausalLM, path: str) -> None:
    """Load trainable-only checkpoint into model."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        # Missing frozen params are expected when saving trainable-only state.
        pass
