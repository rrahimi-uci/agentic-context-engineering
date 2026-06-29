"""Recipe 05 — Persist a playbook and resume from it.

What you'll learn
-----------------
* A playbook is just inspectable data: ``playbook.save(path)`` /
  ``Playbook.load(path)`` round-trip it to JSON.
* How to **warm-start** a fresh ACE engine from a saved playbook so a new process
  begins already knowing what the last one learned — the basis of an agent whose
  memory survives restarts.

Runs fully offline with the deterministic ``SimulatedLLM`` — **no API key**.

    python cookbook/05_save_and_resume.py
"""

from __future__ import annotations

import os
import tempfile

from ace import ACE, ACEConfig, Playbook, SimulatedLLM, TeachingEnvironment, build_teaching_task


def run(path: str | None = None) -> dict:
    cleanup = False
    if path is None:
        fd, path = tempfile.mkstemp(suffix="_playbook.json")
        os.close(fd)
        cleanup = True

    env = TeachingEnvironment(known_fraction=0.3, seed=5)
    task = build_teaching_task(repeats=3, seed=5)
    train, test = task.split(train_frac=0.5, seed=5)

    # --- Session 1: learn and persist ---------------------------------- #
    ace1 = ACE(SimulatedLLM(env), ACEConfig(epochs=5))
    ace1.adapt_offline(train)
    ace1.playbook.save(path)
    learned = len(ace1.playbook)

    # --- Session 2: a brand-new engine warm-started from disk ---------- #
    resumed = Playbook.load(path)
    ace2 = ACE(SimulatedLLM(env), playbook=resumed)
    # No adaptation here — we only evaluate, to show the knowledge persisted.
    result = ace2.evaluate(test)

    if cleanup:
        os.remove(path)

    return {
        "saved_bullets": learned,
        "resumed_bullets": len(resumed),
        "resumed_accuracy": result.accuracy,
    }


def main() -> int:
    r = run()
    print(f"Session 1 learned and saved : {r['saved_bullets']} bullets")
    print(f"Session 2 loaded from disk  : {r['resumed_bullets']} bullets")
    print(f"Session 2 eval accuracy     : {r['resumed_accuracy']:.1f}% (no re-training)")
    print("\n✓ The agent's memory survived a full restart.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
