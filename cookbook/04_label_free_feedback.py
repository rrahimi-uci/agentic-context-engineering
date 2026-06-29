"""Recipe 04 — Learn without ground-truth labels.

What you'll learn
-----------------
* The **label-free** path: instead of handing ACE the correct answer, you hand
  it a *signal* from the environment — did the tests pass? did the API 200 or
  500? — via a ``feedback_fn``.
* This is the realistic setting for agents in production, where you rarely have
  a gold answer but you almost always have an execution outcome.

The extension point
-------------------
    def feedback_fn(sample, generation) -> Feedback:
        ok = run_my_validator(generation.answer)        # YOUR check
        return Feedback(correct=ok, signal="..."  )     # no ground_truth needed

    ace.adapt_online(task, feedback_fn=feedback_fn)

Runs fully offline with the deterministic ``SimulatedLLM`` — **no API key**.

    python cookbook/04_label_free_feedback.py
"""

from __future__ import annotations

from ace import ACE, Feedback, SimulatedLLM, TeachingEnvironment
from ace.tasks import build_teaching_task


def run() -> dict:
    env = TeachingEnvironment(known_fraction=0.3, seed=4)
    task = build_teaching_task(repeats=3, seed=4)

    # An external validator stands in for "tests" or "the environment". It tells
    # us pass/fail and produces a natural-language signal — but NO gold answer is
    # ever passed to ACE as the label to memorize.
    def feedback_fn(sample, generation) -> Feedback:
        passed = task.evaluate(generation.answer, sample)
        return Feedback(
            correct=passed,
            ground_truth=None,  # label-free: we don't reveal the answer
            signal="validator: checks passed" if passed else "validator: checks FAILED",
        )

    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task, feedback_fn=feedback_fn)

    return {
        "accuracy": result.accuracy,
        "playbook_bullets": len(ace.playbook),
        "graded_steps": result.summary()["graded"],
    }


def main() -> int:
    r = run()
    print("Learning purely from execution signals (no gold answers):")
    print(f"  accuracy after adaptation : {r['accuracy']:.1f}%")
    print(f"  playbook bullets learned  : {r['playbook_bullets']}")
    print(f"  graded steps              : {r['graded_steps']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
