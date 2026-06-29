"""Visualization: a live terminal view of a run, and a self-contained HTML report.

* :class:`LiveRunVisualizer` — a ``rich``-powered live dashboard you pass as the
  ``callback`` to :meth:`ACE.adapt_online` / :meth:`ACE.adapt_offline`. It shows
  the playbook growing, running accuracy, and the most recent delta operations in
  real time.
* :func:`render_html_report` — turns one or more :class:`RunResult` objects into
  a single dependency-free HTML file with inline-SVG charts (accuracy &
  playbook-size curves), a delta timeline, and the final playbook. Great for
  READMEs, blog posts, and sharing results.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence

from .engine import RunResult, StepRecord


# --------------------------------------------------------------------------- #
# Live terminal dashboard
# --------------------------------------------------------------------------- #
class LiveRunVisualizer:
    """Animated terminal dashboard for an ACE run.

    Usage::

        with LiveRunVisualizer(title="Online adaptation") as viz:
            result = ace.adapt_online(task, callback=viz)
    """

    def __init__(self, title: str = "ACE Run", total: Optional[int] = None) -> None:
        self.title = title
        self.total = total
        self._records: List[StepRecord] = []
        self._live: Any = None  # rich.live.Live, imported lazily in __enter__

    def __enter__(self) -> "LiveRunVisualizer":
        from rich.live import Live

        self._live = Live(self._render(), refresh_per_second=12, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*exc)

    def __call__(self, record: StepRecord) -> None:
        """Callback target — call on each step."""
        self._records.append(record)
        if self._live is not None:
            self._live.update(self._render())

    # ------------------------------------------------------------------ #
    def _running_accuracy(self) -> float:
        graded = [r for r in self._records if r.correct is not None]
        if not graded:
            return 0.0
        return 100.0 * sum(1 for r in graded if r.correct) / len(graded)

    def _sparkline(self, values: Sequence[float]) -> str:
        if not values:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        lo, hi = min(values), max(values)
        rng = (hi - lo) or 1.0
        return "".join(blocks[int((v - lo) / rng * (len(blocks) - 1))] for v in values[-40:])

    def _render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        latest = self._records[-1] if self._records else None
        acc = self._running_accuracy()
        sizes = [r.playbook_size for r in self._records]

        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        done = len(self._records)
        prog = f"{done}/{self.total}" if self.total else str(done)
        header.add_row(
            f"[bold cyan]{self.title}[/bold cyan]  step [bold]{prog}[/bold]",
            f"[bold]accuracy[/bold] [green]{acc:5.1f}%[/green]",
        )

        stats = Table.grid(expand=True)
        stats.add_column(ratio=1)
        stats.add_column(ratio=1)
        stats.add_column(ratio=1)
        pb_size = latest.playbook_size if latest else 0
        pb_tok = latest.playbook_tokens if latest else 0
        stats.add_row(
            f"playbook bullets: [bold]{pb_size}[/bold]",
            f"≈tokens: [bold]{pb_tok}[/bold]",
            f"phase: [bold]{latest.phase if latest else '-'}[/bold]",
        )

        spark = Table.grid()
        spark.add_row(
            f"playbook growth  [magenta]{self._sparkline([float(s) for s in sizes])}[/magenta]"
        )

        # Recent steps table.
        tbl = Table(expand=True, show_edge=False, pad_edge=False)
        tbl.add_column("#", justify="right", width=4)
        tbl.add_column("ok", width=3)
        tbl.add_column("Δ ops", width=18)
        tbl.add_column("note", overflow="ellipsis")
        for r in self._records[-8:]:
            ok = (
                "[green]✓[/green]" if r.correct else ("[red]✗[/red]" if r.correct is False else "·")
            )
            added = len(r.merge.get("added", [])) if r.merge else 0
            updated = len(r.merge.get("updated", [])) if r.merge else 0
            removed = (
                len(r.refine.get("deduped", [])) + len(r.refine.get("pruned", []))
                if r.refine
                else 0
            )
            collapsed = r.refine.get("collapsed") if r.refine else False
            ops = f"+{added} ~{updated} -{removed}"
            note = "[red bold]COLLAPSE[/red bold]" if collapsed else (r.diagnosis[:50])
            tbl.add_row(str(r.step), ok, ops, note)

        return Panel(Group(header, stats, spark, tbl), border_style="cyan")


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _svg_line_chart(
    series: Dict[str, List[float]], width=560, height=220, ylabel="", colors=None
) -> str:
    colors = colors or ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    pad = 36
    all_vals = [v for s in series.values() for v in s] or [0, 1]
    ymin, ymax = min(all_vals), max(all_vals)
    if ymax == ymin:
        ymax = ymin + 1

    def x(i, n):
        n = max(n - 1, 1)
        return pad + (width - 2 * pad) * i / n

    def y(v):
        return height - pad - (height - 2 * pad) * (v - ymin) / (ymax - ymin)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']
    # axes
    parts.append(
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#cbd5e1"/>'
    )
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#cbd5e1"/>')
    parts.append(
        f'<text x="{pad}" y="{pad - 12}" font-size="11" fill="#64748b">{html.escape(ylabel)}</text>'
    )
    parts.append(
        f'<text x="{pad - 6}" y="{y(ymax) + 4}" font-size="10" fill="#94a3b8" text-anchor="end">{ymax:.0f}</text>'
    )
    parts.append(
        f'<text x="{pad - 6}" y="{y(ymin) + 4}" font-size="10" fill="#94a3b8" text-anchor="end">{ymin:.0f}</text>'
    )
    for ci, (name, s) in enumerate(series.items()):
        color = colors[ci % len(colors)]
        if not s:
            continue
        pts = " ".join(f"{x(i, len(s)):.1f},{y(v):.1f}" for i, v in enumerate(s))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>')
        parts.append(
            f'<circle cx="{x(len(s) - 1, len(s)):.1f}" cy="{y(s[-1]):.1f}" r="3" fill="{color}"/>'
        )
        # legend
        ly = pad + ci * 18
        parts.append(
            f'<rect x="{width - pad - 150}" y="{ly - 9}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{width - pad - 134}" y="{ly + 1}" font-size="11" fill="#334155">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _windowed_accuracy(history: List[StepRecord], window: int = 5) -> List[float]:
    """Rolling accuracy (%) so the curve shows learning over time."""
    out: List[float] = []
    buf: List[int] = []
    for r in history:
        if r.correct is None:
            continue
        buf.append(1 if r.correct else 0)
        w = buf[-window:]
        out.append(100.0 * sum(w) / len(w))
    return out


def render_html_report(
    runs: Dict[str, RunResult],
    title: str = "ACE — Agentic Context Engineering",
    subtitle: str = "Run report",
    out_path: Optional[str] = None,
) -> str:
    """Render one or more named runs into a single self-contained HTML report."""
    acc_series = {name: _windowed_accuracy(r.history) for name, r in runs.items()}
    size_series = {name: [float(s) for s in r.growth_curve] for name, r in runs.items()}

    # Summary cards.
    cards = []
    for name, r in runs.items():
        s = r.summary()
        cards.append(
            f'<div class="card"><h3>{html.escape(name)}</h3>'
            f'<div class="big">{s["accuracy"]:.1f}%</div>'
            f'<div class="muted">accuracy · {s["graded"]} graded</div>'
            f'<div class="muted">playbook: {s["final_playbook_size"]} bullets · ≈{s["final_playbook_tokens"]} tok</div>'
            f"</div>"
        )

    # Final playbook (from the first run that has one).
    pb_html = ""
    primary = next((r for r in runs.values() if r.playbook and len(r.playbook)), None)
    if primary and primary.playbook:
        rows = []
        for b in primary.playbook:
            rows.append(
                f"<tr><td class='mono'>{html.escape(b.id)}</td>"
                f"<td>{html.escape(b.section)}</td>"
                f"<td>{html.escape(b.content)}</td>"
                f"<td>+{b.helpful_count}/-{b.harmful_count}</td></tr>"
            )
        pb_html = (
            "<h2>Final Playbook</h2><table class='pb'>"
            "<tr><th>id</th><th>section</th><th>content</th><th>score</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    # Delta timeline (collapse events highlighted).
    timeline_rows = []
    for name, r in runs.items():
        for rec in r.history:
            collapsed = rec.refine.get("collapsed") if rec.refine else False
            added = len(rec.merge.get("added", [])) if rec.merge else 0
            if added == 0 and not collapsed:
                continue
            cls = "collapse" if collapsed else ""
            label = "CONTEXT COLLAPSE" if collapsed else f"+{added} bullet(s)"
            timeline_rows.append(
                f"<tr class='{cls}'><td>{html.escape(name)}</td><td>{rec.step}</td>"
                f"<td>{'✓' if rec.correct else ('✗' if rec.correct is False else '·')}</td>"
                f"<td>{label}</td><td>{rec.playbook_size}</td></tr>"
            )
    timeline_html = ""
    if timeline_rows:
        timeline_html = (
            "<h2>Adaptation Timeline</h2><table class='pb'>"
            "<tr><th>run</th><th>step</th><th>ok</th><th>event</th><th>playbook size</th></tr>"
            + "".join(timeline_rows[:200])
            + "</table>"
        )

    data_json = json.dumps({n: r.summary() for n, r in runs.items()}, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f8fafc; color: #0f172a; }}
  header {{ background: linear-gradient(135deg,#1e293b,#0f172a); color: #fff; padding: 32px 24px; }}
  header h1 {{ margin: 0 0 4px; font-size: 26px; }}
  header p {{ margin: 0; color: #94a3b8; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 28px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 18px;
          flex: 1 1 180px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card h3 {{ margin: 0 0 6px; font-size: 14px; color: #475569; }}
  .big {{ font-size: 30px; font-weight: 700; color: #2563eb; }}
  .muted {{ color: #64748b; font-size: 12px; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 16px 0; }}
  h2 {{ margin-top: 32px; }}
  table.pb {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden;
             font-size: 13px; border: 1px solid #e2e8f0; }}
  table.pb th, table.pb td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eef2f7; vertical-align: top; }}
  table.pb th {{ background: #f1f5f9; color: #475569; }}
  tr.collapse td {{ background: #fef2f2; color: #b91c1c; font-weight: 600; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #2563eb; }}
  footer {{ text-align: center; color: #94a3b8; padding: 24px; font-size: 12px; }}
  code {{ background:#0f172a; color:#e2e8f0; padding:2px 6px; border-radius:4px; }}
</style></head>
<body>
<header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>
<main>
  <div class="cards">{"".join(cards)}</div>
  <div class="chart"><h2 style="margin-top:0">Accuracy over time (rolling window)</h2>
    {_svg_line_chart(acc_series, ylabel="accuracy %")}</div>
  <div class="chart"><h2 style="margin-top:0">Playbook size over time</h2>
    {_svg_line_chart(size_series, ylabel="bullets")}</div>
  {timeline_html}
  {pb_html}
  <h2>Summary (JSON)</h2>
  <pre style="background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;overflow:auto">{html.escape(data_json)}</pre>
</main>
<footer>Generated by <code>ace.visualize</code> · Agentic Context Engineering</footer>
</body></html>"""


def save_html_report(runs: Dict[str, RunResult], out_path: str, **kwargs) -> str:
    htmltext = render_html_report(runs, out_path=out_path, **kwargs)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(htmltext)
    return out_path
