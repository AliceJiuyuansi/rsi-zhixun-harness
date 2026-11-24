"""C校准参数仅从随镜像交付的固定配置读取，不接触验证标签。"""
import json
import os
from pathlib import Path


def load_calibration_options():
    path = Path(__file__).resolve().parents[2] / "configs/exp/c-paper-dw.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    method = os.environ.get("RSI_C_CALIBRATION_METHOD", config["method"])
    if method not in ("ridge", "paper_dw", "nasa_log"):
        raise ValueError("unknown C calibration method")
    return {"method": method, **{key: config[key] for key in (
        "threshold_cycles", "weight_cap_cycles", "ridge_strength", "correction_limit_cycles", "optimization_steps")}}


def load_c_selection_policy():
    path = Path(__file__).resolve().parents[2] / "configs/exp/c-paper-dw.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return {"rmse_tolerance": float(config["rmse_relative_tolerance"]),
            "min_nasa_gain": float(config["nasa_min_relative_gain"])}
