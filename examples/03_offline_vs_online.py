"""Offline warmup + online adaptation, with cost/latency accounting.

Demonstrates the paper's settings:
  * offline multi-epoch adaptation (build a system-prompt-style playbook), then
  * online test-time adaptation, optionally warm-started from the offline playbook.

Also reports the incremental-update efficiency story: ACE merges deltas with
deterministic, non-LLM logic, so adaptation cost scales with the *delta* size,
not the full context — unlike monolithic rewriting.

Run:  python examples/03_offline_vs_online.py
"""

from ace import ACE, ACEConfig, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.baselines import MonolithicRewriteAgent, StaticAgent

env = TeachingEnvironment(known_fraction=0.3, seed=5)
task = build_teaching_task(repeats=4, seed=5)
train, test = task.split(train_frac=0.4, seed=5)

base = StaticAgent(SimulatedLLM(env)).run(test)

# Online only (cold start).
online_cold = ACE(SimulatedLLM(env)).adapt_online(test)

# Offline warmup, then online (the paper's best AppWorld setting).
warm = ACE(SimulatedLLM(env), ACEConfig(epochs=5))
warm.adapt_offline(train)
warm_eval_before = warm.evaluate(test).accuracy
online_warm = warm.adapt_online(test)

print(f"Base LLM                         : {base.accuracy:5.1f}%")
print(f"ACE online (cold start)          : {online_cold.accuracy:5.1f}%")
print(f"ACE offline→eval (before online) : {warm_eval_before:5.1f}%")
print(f"ACE offline warmup + online      : {online_warm.accuracy:5.1f}%")

# --- Cost story: deltas vs full rewrite (token-ingestion proxy) ----------- #
# Monolithic rewrite re-ingests the whole context each step; ACE ingests only
# the small delta. We approximate ingested tokens to illustrate the gap.
mono = MonolithicRewriteAgent(SimulatedLLM(env)).run(test)
mono_tokens = sum(r.playbook_tokens for r in mono.history)  # re-ingest whole ctx each step
ace_delta_tokens = sum(
    sum(len(op.get("content", "")) // 4 for op in r.delta.get("operations", []))
    for r in online_cold.history
)
reduction = 100 * (1 - ace_delta_tokens / max(mono_tokens, 1))
print("\nAdaptation token-ingestion proxy:")
print(f"  Monolithic (full re-ingest): {mono_tokens:>7} tok")
print(f"  ACE (delta merge only)     : {ace_delta_tokens:>7} tok   (-{reduction:.1f}%)")
