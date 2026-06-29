"""Recipe 07 — Inspect the playbook and generate an HTML report.

What you'll learn
-----------------
* How to introspect what the agent learned: iterate bullets (id, section,
  helpful/harmful score), read :meth:`~ace.playbook.Playbook.stats`, and estimate
  token cost — the playbook is fully interpretable and editable.
* How to render a polished, dependency-free **HTML report** of a run with
  :func:`~ace.visualize.save_html_report` (accuracy & growth charts, a delta
  timeline, and the final playbook).

Runs fully offline with the deterministic ``SimulatedLLM`` — **no API key**.

    python cookbook/07_inspect_and_report.py        # writes cookbook_report.html
"""

from __future__ import annotations

import os
import tempfile

from ace import ACE, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.visualize import save_html_report


def run(report_path: str | None = None) -> dict:
    cleanup = False
    if report_path is None:
        fd, report_path = tempfile.mkstemp(suffix="_ace_report.html")
        os.close(fd)
        cleanup = True

    env = TeachingEnvironment(known_fraction=0.3, seed=7)
    task = build_teaching_task(repeats=3, seed=7)

    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task)

    # Inspect: every bullet is addressable and scored.
    bullets = [
        {"id": b.id, "section": b.section, "score": b.score, "content": b.content}
        for b in ace.playbook
    ]
    stats = ace.playbook.stats()

    # Render a shareable HTML report of the whole run.
    save_html_report(
        {"ACE (online)": result},
        out_path=report_path,
        subtitle="Cookbook · online adaptation on the teaching task",
    )
    report_html = ""
    with open(report_path, encoding="utf-8") as f:
        report_html = f.read()

    if cleanup:
        os.remove(report_path)

    return {
        "num_bullets": stats["num_bullets"],
        "approx_tokens": stats["approx_tokens"],
        "sections": stats["sections"],
        "bullets": bullets,
        "report_path": report_path,
        "report_bytes": len(report_html),
        "report_is_html": report_html.lstrip().startswith("<!DOCTYPE html>"),
    }


def main() -> int:
    report_path = "cookbook_report.html"
    r = run(report_path=report_path)
    print(f"Playbook: {r['num_bullets']} bullets, ≈{r['approx_tokens']} tokens")
    print("Per-section counts:", r["sections"])
    print("\nTop bullets by score:")
    for b in sorted(r["bullets"], key=lambda x: -x["score"])[:5]:
        print(f"  [{b['id']}] (+score {b['score']}) {b['content']}")
    print(f"\nHTML report written to {report_path} ({r['report_bytes']} bytes).")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
