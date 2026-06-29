"""Recipe 03 — Bring your own task.

What you'll learn
-----------------
* How to define a :class:`~ace.tasks.Task` from your own :class:`~ace.tasks.Sample`
  data with your own ``evaluate`` scorer.
* That ACE is domain-agnostic: the engine never assumes anything about your data
  beyond "here is a question, here is how to score an answer".

The shape you implement
------------------------
    Sample(id=..., question=..., answer=...)          # your data points
    Task(name=..., samples=[...], evaluate=scorer)    # + how to grade an answer

Swap ``SimulatedLLM`` for ``OpenAILLM(model="gpt-4o-mini")`` and the *exact same
code* runs against a real model. We use the simulator here so the recipe is
deterministic and needs **no API key**.

    python cookbook/03_your_own_task.py
"""

from __future__ import annotations

from ace import ACE, SimulatedLLM, Sample, Task, TeachingEnvironment
from ace.tasks import build_teaching_task


def my_scorer(prediction: str, sample: Sample) -> bool:
    """Your grading logic. Here: case-insensitive containment of the gold answer."""
    gold = sample.answer.strip().lower()
    pred = (prediction or "").strip().lower()
    return gold == pred or (len(pred) > 3 and gold.split(")")[-1].strip() in pred)


def build_my_task() -> Task:
    # In a real project these Samples come from your own dataset. Here we borrow
    # the simulator's items so the recipe can show measurable learning offline.
    source = build_teaching_task(repeats=3, seed=3)
    samples = [
        Sample(
            id=s.id,
            question=s.question,
            answer=s.answer,
            concept=s.concept,          # the simulator's hidden rule key
            rule_text=s.rule_text,
            metadata=s.metadata,
        )
        for s in source.samples
    ]
    return Task(name="my-domain", samples=samples, evaluate=my_scorer,
                description="A task defined entirely from my own data + scorer.")


def run() -> dict:
    env = TeachingEnvironment(known_fraction=0.3, seed=3)
    task = build_my_task()

    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task)

    return {
        "task_name": task.name,
        "num_samples": len(task.samples),
        "accuracy": result.accuracy,
        "playbook_bullets": len(ace.playbook),
    }


def main() -> int:
    r = run()
    print(f"Task '{r['task_name']}' with {r['num_samples']} of your own samples.")
    print(f"Accuracy after online adaptation : {r['accuracy']:.1f}%")
    print(f"Playbook learned                 : {r['playbook_bullets']} bullets")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
