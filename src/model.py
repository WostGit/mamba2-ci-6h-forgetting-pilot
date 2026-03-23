"""Model loading and checkpoint IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(cfg: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Mamba-2 checkpoint in CPU mode."""
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cpu")
    model.train()
    return model, tokenizer


def configure_trainable_parameters(model: torch.nn.Module, cfg: Dict[str, Any]) -> int:
    """Enable either full fine-tuning or lightweight fallback subset."""
    if cfg["train"]["full_finetune"]:
        for p in model.parameters():
            p.requires_grad = True
    else:
        keys = tuple(cfg["train"]["trainable_param_keywords"])
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in keys)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, out_dir: str, name: str) -> Path:
    """Save trainable tensors only to keep artifacts small."""
    p = Path(out_dir) / "checkpoints" / name
    p.parent.mkdir(parents=True, exist_ok=True)

    trainable_keys = {n for n, param in model.named_parameters() if param.requires_grad}
    payload = {
        "trainable_keys": sorted(trainable_keys),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable_keys},
    }
    torch.save(payload, p)
    return p


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Load state dict from local checkpoint if present."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
