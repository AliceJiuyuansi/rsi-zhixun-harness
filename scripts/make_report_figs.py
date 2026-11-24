"""Generate the two small SVG figures used by the v2 technical report.

Pure Python (no matplotlib); the SVGs are text files and stay reviewable in git.

    python scripts/make_report_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figs"
TRIAL = ROOT / "_state" / "trial_report_completed.json"
REPLAY = ROOT / "reports" / "v2_replay_v1.json"

W, H = 760, 360
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 40, 60


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _axes(max_value: float, ylabel: str) -> list[str]:
    width, height = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    parts = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    for tick in range(6):
        value = max_value * tick / 5
        y = PAD_T + height - height * tick / 5
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
                     f'stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-family="Arial" font-size="11" fill="#666">{value:.3g}</text>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" '
                 f'stroke="#888" stroke-width="1"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" '
                 f'stroke="#888" stroke-width="1"/>')
    parts.append(f'<text x="18" y="{H / 2}" transform="rotate(-90 18 {H / 2})" '
                 f'text-anchor="middle" font-family="Arial" font-size="12" fill="#444">{_esc(ylabel)}</text>')
    return parts


def bar_chart(path: Path, groups: list[tuple[str, float, float]], ylabel: str,
              title: str, legend=("v1", "v2")) -> None:
    max_value = max(max(a, b) for _, a, b in groups) * 1.15 or 1
    height = H - PAD_T - PAD_B
    width = W - PAD_L - PAD_R
    slot = width / max(len(groups), 1)
    bar_w = slot * 0.24
    parts = _axes(max_value, ylabel)
    colors = ("#9aa5b1", "#2f6fdb")
    for index, (label, first, second) in enumerate(groups):
        base_x = PAD_L + slot * index + slot * 0.5
        for offset, value, color in ((-bar_w / 2, first, colors[0]), (bar_w / 2, second, colors[1])):
            bar_h = height * value / max_value
            x = base_x + offset - bar_w / 2
            y = H - PAD_B - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                         f'fill="{color}" rx="2"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                         f'font-family="Arial" font-size="10" fill="#333">{value:.0f}</text>')
        parts.append(f'<text x="{base_x:.1f}" y="{H - PAD_B + 20}" text-anchor="middle" '
                     f'font-family="Arial" font-size="12" fill="#333">{_esc(label)}</text>')
    parts.append(f'<text x="{PAD_L}" y="22" font-family="Arial" font-size="15" fill="#222">{_esc(title)}</text>')
    for index, (name, color) in enumerate(zip(legend, colors)):
        x = W - PAD_R - 150 + index * 80
        parts.append(f'<rect x="{x}" y="12" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{x + 16}" y="23" font-family="Arial" font-size="12" fill="#333">{name}</text>')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    svg += parts + ["</svg>"]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def line_chart(path: Path, points: list[tuple[int, float, str]], ylabel: str, title: str) -> None:
    if not points:
        return
    low = min(value for _, value, _ in points)
    high = max(value for _, value, _ in points)
    span = max(high - low, 1e-6)
    low -= span * 0.08
    high += span * 0.08
    width = W - PAD_L - PAD_R
    height = H - PAD_T - PAD_B
    parts = _axes(high, ylabel)
    # re-label the y ticks from low..high instead of 0..max
    parts = [part for part in parts if "text-anchor=\"end\"" not in part]
    parts = _axes(high, ylabel)[:1]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = PAD_T + height - height * tick / 5
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
                     f'stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-family="Arial" font-size="11" fill="#666">{value:.3f}</text>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" stroke="#888"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" stroke="#888"/>')

    coords = []
    for index, (round_index, value, stage) in enumerate(points):
        x = PAD_L + width * index / max(len(points) - 1, 1)
        y = PAD_T + height - height * (value - low) / (high - low)
        coords.append((x, y))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2f6fdb"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - PAD_B + 20}" text-anchor="middle" '
                     f'font-family="Arial" font-size="10" fill="#333">{round_index}</text>')
        if index == 0 or points[index - 1][2] != stage:
            parts.append(f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H - PAD_B}" '
                         f'stroke="#f0c36d" stroke-width="1" stroke-dasharray="3 3"/>')
            parts.append(f'<text x="{x + 3:.1f}" y="{PAD_T + 12}" font-family="Arial" '
                         f'font-size="9" fill="#a57516" transform="rotate(-90 {x + 3:.1f} {PAD_T + 12})">'
                         f'{_esc(stage)}</text>')
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2f6fdb" stroke-width="2"/>')
    parts.append(f'<text x="{PAD_L}" y="22" font-family="Arial" font-size="15" fill="#222">{_esc(title)}</text>')
    parts.append(f'<text x="{W / 2}" y="{H - 12}" text-anchor="middle" font-family="Arial" '
                 f'font-size="12" fill="#444">round</text>')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    svg += parts + ["</svg>"]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    groups = [(item["dataset"], item["v1_epochs"], item["v2_epochs"])
              for item in replay["datasets"]]
    bar_chart(FIG / "v1_v2_epochs.svg", groups, "epochs per dataset (A/B/C)",
              "Training budget: v1 vs v2 replay")

    trial = json.loads(TRIAL.read_text(encoding="utf-8"))
    dataset_a = next(item for item in trial["report"]["workspace_report"]["datasets"]
                     if item["key"] == "A")
    points = []
    for record in dataset_a["rounds"]:
        candidates = record.get("candidates") or []
        metrics = candidates[0].get("validation_metrics") if candidates else record.get("validation_metrics")
        if metrics and "macro_f1" in metrics:
            points.append((int(record["round_index"]), float(metrics["macro_f1"]),
                           record.get("stage_id", "")))
    line_chart(FIG / "a_round_f1.svg", points, "macro-F1 (public validation)",
               "A task: validation macro-F1 per round (v1)")
    print("wrote", FIG / "v1_v2_epochs.svg")
    print("wrote", FIG / "a_round_f1.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
