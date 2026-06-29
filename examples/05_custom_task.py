"""Bring your own task + your own feedback — the general-purpose path.

This shows the two extension points that make ACE usable on *any* problem:

  1. Define a `Task` from your own data with your own `evaluate` scorer.
  2. Pass a `feedback_fn(sample, generation) -> Feedback` to supply custom or
     label-free feedback (execution signals, a reward function, an LLM judge).

Runs fully offline (no API key) by using the bundled SimulatedLLM, but the
exact same code works with `OpenAILLM(model=...)` for real models.

Run:  python examples/05_custom_task.py
"""

from ace import ACE, Feedback, Sample, SimulatedLLM, Task, TeachingEnvironment

# 1) Build a Task from your own samples. (Here we borrow the teaching env's
#    rules as "your data"; in practice these are your prompts/answers.)
env = TeachingEnvironment(known_fraction=0.2, seed=11)
from ace.tasks import build_teaching_task

base_task = build_teaching_task(repeats=3, seed=11)
my_task = Task(
    name="my-domain",
    samples=base_task.samples,
    evaluate=base_task.evaluate,  # your own scorer
    description="A custom domain task.",
)


# 2) Supply feedback yourself — e.g. a reward function or environment signal.
#    No ground-truth labels are handed to ACE directly; we compute a signal.
def my_feedback(sample: Sample, generation) -> Feedback:
    passed = my_task.evaluate(generation.answer, sample)
    return Feedback(
        correct=passed,
        ground_truth=sample.answer,  # optional; omit for the pure label-free path
        signal="unit tests passed" if passed else "unit tests FAILED",
    )


ace = ACE(SimulatedLLM(env))
result = ace.adapt_online(my_task, feedback_fn=my_feedback)

print(f"Final accuracy : {result.accuracy:.1f}%")
print(f"Playbook       : {len(ace.playbook)} bullets learned from your feedback\n")
print(ace.playbook.render())
