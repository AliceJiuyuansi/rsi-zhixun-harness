"""Audit log (受控 RSI 五件套 #5, docs/04 §4).

Append-only JSONL, one line per notable event, flushed immediately so a crash
still leaves evidence. Fields follow docs/04: time, trigger signal, diagnosis,
proposal, gate reason, cost, result.

The log is fail-open on purpose: a broken audit path must never kill a run
(that would violate "runs unattended to completion"), but it is reported once.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass
    return str(obj)


class AuditLog:
    def __init__(self, path: str | Path | None = None, run: str = "",
                 enabled: bool = True):
        self.path = Path(path) if path else None
        self.run = run
        self.enabled = bool(enabled) and self.path is not None
        self.count = 0
        self.events: list[str] = []
        self._broken = False
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text("", encoding="utf-8")  # fresh run, append-only after
            except OSError:
                self.enabled = False

    def write(self, event: str, **fields) -> None:
        if not self.enabled or self._broken:
            return
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run": self.run,
            "event": str(event),
        }
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=_default)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self.count += 1
            self.events.append(str(event))
        except (OSError, TypeError, ValueError) as exc:
            self._broken = True
            print(f"[audit] disabled after write failure: {exc}")

    def as_dict(self) -> dict:
        return {"path": str(self.path) if self.path else "",
                "enabled": self.enabled,
                "records": self.count}
