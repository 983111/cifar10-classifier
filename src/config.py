from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def apply_cli_overrides(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    for key, value in overrides.items():
        if value is not None:
            OmegaConf.update(cfg, key, value, merge=False)
    return cfg
