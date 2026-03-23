"""Model loading and checkpoint I/O utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, cache_dir: str, dtype: str = "float32"):
    """Load tokenizer + causal LM for CPU-only execution."""
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    )
    model.to("cpu")
    return model, tokenizer


def apply_parameter_efficient_mode(model, mode: str = "lm_head_only") -> int:
    """Freeze all parameters except a tiny trainable subset for CPU feasibility."""
    for p in model.parameters():
        p.requires_grad = False

    trainable = 0
    if mode == "lm_head_only":
        head = getattr(model, "lm_head", None)
        if head is None:
            raise ValueError("Model has no lm_head; cannot apply lm_head_only fallback.")
        for p in head.parameters():
            p.requires_grad = True
            trainable += p.numel()
    elif mode == "last_block_and_head":
        blocks = getattr(model, "backbone", None)
        layers = getattr(blocks, "layers", None)
        if layers is None:
            raise ValueError("Could not locate model backbone.layers.")
        for p in layers[-1].parameters():
            p.requires_grad = True
            trainable += p.numel()
        for p in model.lm_head.parameters():
            p.requires_grad = True
            trainable += p.numel()
    else:
        raise ValueError(f"Unsupported trainable mode: {mode}")
    return trainable


def save_checkpoint(model, out_dir: str | Path) -> None:
    """Persist model weights to a local checkpoint directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)


def load_local_checkpoint(model_dir: str | Path, cache_dir: str):
    """Load a previously saved local checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), cache_dir=cache_dir)
    model.to("cpu")
    return model
