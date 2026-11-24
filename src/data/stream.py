"""Time-ordered stream access: batches, warmup window, delayed-label history."""
from __future__ import annotations

import numpy as np


class Stream:
    def __init__(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        self.X, self.y, self.t = X, y, t
        self.steps = np.unique(t)
        self._idx = {int(s): np.where(t == s)[0] for s in self.steps}

    def batch(self, step: int):
        idx = self._idx[int(step)]
        return self.X[idx], self.y[idx]

    def concat_steps(self, steps) -> tuple[np.ndarray, np.ndarray]:
        idx = np.concatenate([self._idx[int(s)] for s in steps]) if len(steps) else np.array([], dtype=int)
        return self.X[idx], self.y[idx]

    def warmup(self, n_steps: int):
        return self.concat_steps(self.steps[:n_steps])

    def recent(self, upto_step: int, window_steps: int = 40) -> np.ndarray:
        """Unlabeled features over the last `window_steps` steps.

        Drift statistics (PSI) need a real sample size; a single step batch of a
        few rows is far too small for a 10-bin histogram.
        """
        last = int(upto_step)
        first = max(int(self.steps[0]), last - int(window_steps) + 1)
        steps = [s for s in self.steps if first <= s <= last]
        idx = (np.concatenate([self._idx[int(s)] for s in steps])
               if steps else np.array([], dtype=int))
        return self.X[idx]

    def history(self, upto_step: int, label_lag: int = 1, window_steps: int = 40):
        """Labeled data available at time `upto_step` (labels lag by `label_lag`)."""
        last = int(upto_step) - int(label_lag)
        first = max(int(self.steps[0]), last - int(window_steps) + 1)
        steps = [s for s in self.steps if first <= s <= last]
        return self.concat_steps(steps)
