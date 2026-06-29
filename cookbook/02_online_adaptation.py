"""Recipe 02 — Learn at test time with online adaptation.

What you'll learn
-----------------
* The **online** regime: for every sample the agent *predicts first*, then
  *learns* from feedback — adapting as it goes, with no separate training phase.
* How to read the run history to *prove* the agent is improving (compare the
  accuracy of the first third of the run against the last third).

When to use online vs offline
------------------------------
* **Online** — a live stream of tasks; you want the agent to get better mid-flight.
* **Offline** (Recipe 01) — you have a training set and want a strong playbook
  before deployment. The two compose: warm-start online from an offline playbook.

Runs fully offline with the deterministic ``SimulatedLLM`` — **no API key**.

    python cookbook/02_online_adaptation.py
"""

from __future__ import annotations

from ace import ACE, SimulatedLLM, TeachingEnvironment, build_teaching_task


def run() -> dict:
    env = TeachingEnvironment(known_fraction=0.30, seed=2)
    task = build_teaching_task(repeats=3, seed=2)

    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task)  # predict → learn, sample by sample

    n = len(result.history)
    early = result.accuracy_in_window(0, n // 3)          # before much learning
    late = result.accuracy_in_window(2 * n // 3, n)       # after accumulating rules

    return {
        "overall_accuracy": result.accuracy,
        "early_accuracy": early,
        "late_accuracy": late,
        "playbook_bullets": len(ace.playbook),
        "growth_curve": result.growth_curve,
    }


def main() -> int:
    r = run()
    print(f"Overall accuracy across the run : {r['overall_accuracy']:5.1f}%")
    print(f"  first third of the stream     : {r['early_accuracy']:5.1f}%")
    print(f"  last third  of the stream     : {r['late_accuracy']:5.1f}%")
    print(f"\nPlaybook grew to {r['playbook_bullets']} bullets.")
    print("Growth curve (bullets per step):", r["growth_curve"])
    if r["late_accuracy"] >= r["early_accuracy"]:
        print("\n✓ The agent measurably improved as it adapted online.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
