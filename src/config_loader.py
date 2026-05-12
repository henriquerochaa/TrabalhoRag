from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# resolve config.yaml a partir da raiz do projeto, independente do cwd
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
