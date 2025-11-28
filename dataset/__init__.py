"""Path helpers for dataset assets used across the project."""

from __future__ import annotations

from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SFT_DIR = PROCESSED_DIR / "sft"
RL_DIR = PROCESSED_DIR / "rl"
EVAL_DIR = PROCESSED_DIR / "eval"


def ensure_structure() -> None:
    """Create the expected on-disk layout if it does not exist yet."""

    for path in (RAW_DIR, INTERIM_DIR, SFT_DIR, RL_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
