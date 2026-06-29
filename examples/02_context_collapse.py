"""Reproduce *context collapse* (Figure 2) and show ACE avoids it.

Compares, on the same teaching environment and feedback:
  * Base LLM (no adaptation)
  * Monolithic rewrite (re-writes the whole context each step → collapses)
  * ACE (incremental delta updates → no collapse)

Writes an HTML report with accuracy + playbook-size curves.

Run:  python examples/02_context_collapse.py
"""

from ace import ACE, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.baselines import MonolithicRewriteAgent, StaticAgent
from ace.visualize import save_html_report

env = TeachingEnvironment(known_fraction=0.3, seed=3)
task = build_teaching_task(repeats=5, seed=3)

base = StaticAgent(SimulatedLLM(env)).run(task)
mono = MonolithicRewriteAgent(SimulatedLLM(env), collapse_prob=0.2).run(task)
ace = ACE(SimulatedLLM(env)).adapt_online(task)

collapses = sum(1 for r in mono.history if r.refine.get("collapsed"))

print(f"{'Method':<28}{'Accuracy':>10}{'Final ctx':>12}")
print("-" * 50)
print(f"{'Base LLM (no context)':<28}{base.accuracy:>9.1f}%{0:>12}")
print(
    f"{'Monolithic rewrite':<28}{mono.accuracy:>9.1f}%{mono.history[-1].playbook_size:>12}"
    f"   ({collapses} collapses)"
)
print(f"{'ACE (incremental)':<28}{ace.accuracy:>9.1f}%{len(ace.playbook):>12}   (no collapse)")

path = save_html_report(
    {"ACE (incremental)": ace, "Monolithic rewrite": mono},
    out_path="ace_report.html",
    subtitle="Context collapse: monolithic rewrite vs ACE incremental deltas",
)
print(f"\nHTML report: {path}")
