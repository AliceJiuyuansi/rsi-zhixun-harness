"""Cost ledger: compute seconds + labels + tokens, per run and cumulative."""
from __future__ import annotations

from dataclasses import asdict

from src.schema import CostRecord


class CostLedger:
    def __init__(self, alpha_gpu: float = 1.0, beta_label: float = 0.001, gamma_token: float = 0.0):
        self.alpha, self.beta, self.gamma = alpha_gpu, beta_label, gamma_token
        self.total = CostRecord()
        self.train_seconds = 0.0
        self.infer_seconds = 0.0
        self.retrains = 0
        self.rollbacks = 0

    def add_train(self, seconds: float, n_labels: int, is_retrain: bool = True) -> None:
        self.train_seconds += seconds
        self.total.gpu_seconds += seconds
        self.total.labels += int(n_labels)
        self.total.wall_seconds += seconds
        if is_retrain:
            self.retrains += 1

    def add_inference(self, seconds: float) -> None:
        self.infer_seconds += seconds
        self.total.gpu_seconds += seconds
        self.total.wall_seconds += seconds

    def add_tokens(self, n: int) -> None:
        self.total.tokens += int(n)

    def cost(self) -> float:
        return (self.alpha * self.total.gpu_seconds
                + self.beta * self.total.labels
                + self.gamma * self.total.tokens)

    def unit_gain_cost(self, delta_perf: float, eps: float = 1e-3) -> float:
        return self.cost() / max(float(delta_perf), eps)

    def as_dict(self) -> dict:
        d = asdict(self.total)
        d.update({
            "train_seconds": self.train_seconds,
            "infer_seconds": self.infer_seconds,
            "retrains": self.retrains,
            "rollbacks": self.rollbacks,
            "weighted_cost": self.cost(),
        })
        return d
