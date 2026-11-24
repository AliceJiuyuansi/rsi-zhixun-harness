"""XJTU-SY bearing run-to-failure dataset -> compact feature cache.

Raw layout (after extraction):
    <root>/35Hz12kN/Bearing1_1/1.csv
    <root>/37.5Hz11kN/Bearing2_1/1.csv
    <root>/40Hz10kN/Bearing3_1/1.csv

Each CSV is 32768 rows x 2 channels (horizontal, vertical), sampled at 25.6 kHz.
We compute 12 time-domain + 6 frequency-domain features per channel -> 36 features per snapshot.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FS = 25600.0
N_SAMPLES = 32768
FEATURE_NAMES: list[str] = []

_TIME_FEATS = ["mean", "std", "rms", "max_abs", "p2p", "skew", "kurtosis",
               "crest", "shape", "impulse", "clearance", "zcr"]
_FREQ_FEATS = ["band0", "band1", "band2", "band3", "centroid", "spread"]


def _feature_names() -> list[str]:
    names = []
    for ch in ("h", "v"):
        names += [f"{ch}_{f}" for f in _TIME_FEATS + _FREQ_FEATS]
    return names


FEATURE_NAMES = _feature_names()


def _time_features(x: np.ndarray) -> list[float]:
    x = x.astype(np.float64)
    mean = float(x.mean())
    z = x - mean
    std = float(z.std()) or 1e-12
    rms = float(np.sqrt(np.mean(z * z))) or 1e-12
    max_abs = float(np.max(np.abs(z)))
    mean_abs = float(np.mean(np.abs(z))) or 1e-12
    kurt = float(np.mean(z ** 4) / (rms ** 4))
    skew = float(np.mean(z ** 3) / (std ** 3))
    crest = max_abs / rms
    shape = rms / mean_abs
    impulse = max_abs / mean_abs
    clearance = max_abs / (float(np.mean(np.sqrt(np.abs(z)))) ** 2 or 1e-12)
    zcr = float(np.mean(np.diff(np.signbit(z).astype(np.int8)) != 0))
    return [mean, std, rms, max_abs, float(z.max() - z.min()), skew, kurt,
            crest, shape, impulse, clearance, zcr]


def _freq_features(x: np.ndarray) -> list[float]:
    n = len(x)
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    total = float(spec.sum()) or 1e-12
    p = spec / total
    edges = [0.0, 2000.0, 5000.0, 8000.0, FS / 2]
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (freqs >= a) & (freqs < b)
        bands.append(float(p[m].sum()))
    centroid = float(np.sum(freqs * p))
    spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * p)))
    return bands + [centroid, spread]


def extract_features(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    h = _time_features(arr[:, 0]) + _freq_features(arr[:, 0])
    v = _time_features(arr[:, 1]) + _freq_features(arr[:, 1])
    return np.asarray(h + v, dtype=np.float32)


def _snapshot_id(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def bearing_files(bearing_dir: str | Path) -> list[Path]:
    files = [p for p in Path(bearing_dir).glob("*.csv")]
    return sorted(files, key=_snapshot_id)


def bearing_features(bearing_dir: str | Path, limit: int | None = None) -> dict:
    files = bearing_files(bearing_dir)
    if limit:
        files = files[:limit]
    X = np.zeros((len(files), len(FEATURE_NAMES)), dtype=np.float32)
    ids = np.zeros(len(files), dtype=np.int32)
    for i, f in enumerate(files):
        X[i] = extract_features(f)
        ids[i] = _snapshot_id(f)
    n = len(files)
    rul = (n - 1 - np.arange(n, dtype=np.float32)) / max(n - 1, 1)
    return {
        "X": X,
        "rul": rul,
        "file_id": ids,
        "bearing": Path(bearing_dir).name,
        "condition": Path(bearing_dir).parent.name,
        "n": n,
    }


def discover_bearings(root: str | Path) -> list[tuple[str, Path]]:
    root = Path(root)
    out = []
    for cond in sorted([p for p in root.iterdir() if p.is_dir()]):
        for b in sorted([p for p in cond.iterdir() if p.is_dir()]):
            if any(b.glob("*.csv")):
                out.append((f"{cond.name}/{b.name}", b))
    return out
