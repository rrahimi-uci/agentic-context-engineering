"""Recipe 01 — Build your first self-improving playbook (offline).

What you'll learn
-----------------
* The three-line ACE workflow: **construct → adapt_offline → evaluate**.
* How a base LLM with *no* context compares to ACE after it has built a playbook.

Why offline?
------------
Offline adaptation revisits a training split across several epochs to distil a
strong, reusable playbook — think of it as *optimizing the system prompt* once,
up front. You then evaluate on held-out data the playbook has never seen.

Runs fully offline with the deterministic ``SimulatedLLM`` — **no API key**.

    python cookbook/01_first_playbook.py
"""

from __future__ import annotations

from ace import ACE, ACEConfig, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.baselines import StaticAgent


def run() -> dict:
    """Build a playbook offline and measure the lift on held-out data."""
    env = TeachingEnvironment(known_fraction=0.35, seed=1)
    task = build_teaching_task(repeats=3, seed=1)
    train, test = task.split(train_frac=0.5, seed=1)

    # 1) Baseline: the same model with no learned context.
    base = StaticAgent(SimulatedLLM(env)).run(test)

    # 2) ACE: build an evolving playbook on the train split, then evaluate.
    ace = ACE(SimulatedLLM(env), ACEConfig(epochs=5))
    ace.adapt_offline(train)
    adapted = ace.evaluate(test)

    return {
        "base_accuracy": base.accuracy,
        "ace_accuracy": adapted.accuracy,
        "playbook_bullets": len(ace.playbook),
        "playbook": ace.playbook.render(),
    }


def main() -> int:
    r = run()
    lift = r["ace_accuracy"] - r["base_accuracy"]
    print("Base LLM (no playbook) :", f"{r['base_accuracy']:5.1f}%")
    print("ACE  (offline-adapted) :", f"{r['ace_accuracy']:5.1f}%", f"(+{lift:.1f} pts)")
    print(f"\nThe agent wrote itself a {r['playbook_bullets']}-bullet playbook:\n")
    print(r["playbook"])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
