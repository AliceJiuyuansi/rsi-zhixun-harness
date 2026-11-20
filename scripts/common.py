from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "configs"
EXP_DIR = ROOT / "experiments"


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(stream_name: str) -> dict:
    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    path = CONFIG_DIR / "streams" / f"{stream_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"stream config not found: {path}")
    stream_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = deep_merge(base, stream_cfg)
    cfg.setdefault("data", {})
    cfg["data"]["drift"] = (stream_cfg.get("data") or {}).get("drift", {"kind": "none"})
    cfg["stream_name"] = stream_name
    return cfg


def new_outdir(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = EXP_DIR / f"{stamp}-{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_streams() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "streams").glob("*.yaml"))
