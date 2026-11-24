"""SwanLab wrapper that never blocks a run.

- auto: use online if credentials exist, otherwise local
- local: write to ./swanlog (no login needed)
- online: upload to the team workspace
- disabled: no-op (unit tests / CI without the SDK)
"""
from __future__ import annotations

import os
from pathlib import Path


class RunLogger:
    def __init__(self, run=None, mode: str = "disabled"):
        self.run = run
        self.mode = mode
        self._swan = None if run is None else __import__("swanlab")

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self._swan.log(data, step=step)
        except Exception as exc:  # logging must never kill training
            print(f"[swan] log failed: {exc}")

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self._swan.finish()
        except Exception as exc:
            print(f"[swan] finish failed: {exc}")

    @property
    def url(self) -> str:
        if self.run is None:
            return ""
        for attr in ("url", "cloud_url"):
            try:
                val = getattr(self.run, attr, None)
            except Exception:
                # local/offline runs raise when asked for a cloud URL
                return ""
            if val:
                return str(val)
        return ""


def _has_credentials() -> bool:
    if os.environ.get("SWANLAB_API_KEY"):
        return True
    for name in (".netrc", "_netrc"):
        p = Path.home() / name
        try:
            if p.exists() and "swanlab" in p.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            pass
    return False


def start_run(project: str, workspace: str, name: str, config: dict,
              outdir: str, tags=None, group: str | None = None,
              mode: str = "auto") -> RunLogger:
    mode = (mode or "auto").lower()
    if mode == "auto":
        mode = "online" if _has_credentials() else "local"
    if mode == "disabled":
        return RunLogger(None, "disabled")

    try:
        import swanlab
        Path(outdir).mkdir(parents=True, exist_ok=True)
        run = swanlab.init(
            project=project,
            workspace=workspace,
            name=name,
            description=f"{group or ''} | {name}",
            config=config,
            mode=mode,
            log_dir=str(Path(outdir) / "swanlog"),
            tags=tags or [],
            group=group,
            on_online_error="offline",
        )
        print(f"[swan] mode={mode} name={name}")
        return RunLogger(run, mode)
    except Exception as exc:
        print(f"[swan] init failed, running without logging: {exc}")
        return RunLogger(None, mode)
