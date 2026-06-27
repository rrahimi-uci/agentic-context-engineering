"""Quickstart: ACE in ~10 lines, no API key required.

Run:  python examples/01_quickstart.py
"""

from ace import ACE, ACEConfig, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.baselines import StaticAgent

env = TeachingEnvironment(known_fraction=0.35, seed=1)
task = build_teaching_task(repeats=3, seed=1)
train, test = task.split(train_frac=0.5, seed=1)

# 1) Base model with no learned context.
base = StaticAgent(SimulatedLLM(env)).run(test)

# 2) ACE: build an evolving playbook offline, then evaluate on held-out data.
ace = ACE(SimulatedLLM(env), ACEConfig(epochs=5))
ace.adapt_offline(train)
result = ace.evaluate(test)

print(f"Base LLM accuracy : {base.accuracy:5.1f}%")
print(f"ACE   accuracy    : {result.accuracy:5.1f}%   (+{result.accuracy - base.accuracy:.1f} pts)")
print(f"Playbook learned  : {len(ace.playbook)} bullets, ~{ace.playbook.approx_tokens()} tokens\n")
print(ace.playbook.render())
