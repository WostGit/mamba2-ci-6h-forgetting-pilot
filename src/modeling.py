"""Model loading and lightweight checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXTokenizerFast


def load_model_and_tokenizer(cfg: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Mamba-2 model and tokenizer in CPU mode.

    Runtime-saving and robustness choice: bypass AutoTokenizer for this model.
    The checkpoint advertises a GPT-NeoX-style tokenizer, and the auto path is
    flaky in CI due to a broken slow->fast conversion branch.
    """
    name = cfg["model"]["name"]
    tokenizer_name = cfg.get("model", {}).get("tokenizer_name", "EleutherAI/gpt-neox-20b")
    tok = GPTNeoXTokenizerFast.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
