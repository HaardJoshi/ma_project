"""
Configuration loader.

Loads a YAML config, merges it with default.yaml, and resolves
all paths relative to the project root.
"""

import os
import yaml
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (base is mutated)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | None = None) -> dict:
    """
    Load experiment config.

    1. Reads ``configs/default.yaml`` as the base.
    2. If *config_path* is given, deep-merges it on top.
    3. Resolves relative paths to absolute using PROJECT_ROOT.

    Returns
    -------
    dict
        Fully merged configuration dictionary.
    """
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        p = Path(config_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        with open(p, "r") as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)

    # Resolve path keys to absolute paths
    for section in ("data", "output"):
        if section in cfg:
            for key, val in cfg[section].items():
                if isinstance(val, str) and ("/" in val or val.startswith(".")):
                    resolved = PROJECT_ROOT / val
                    cfg[section][key] = str(resolved)

    return cfg


def get_device(cfg: dict) -> str:
    """Return the torch device string from config ('auto' → best available)."""
    device = cfg.get("training", {}).get("device", "auto")
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
