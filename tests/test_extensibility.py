"""Tests for the generality hooks: custom feedback_fn and LLM-driven Curator."""

from ace import ACE, ACEConfig, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.delta import DeltaOp
from ace.feedback import Feedback
from ace.playbook import Playbook
from ace.roles import Curator, Generation, Reflection
from ace.tasks import Sample


# --------------------------------------------------------------------------- #
# #1  feedback_fn hook
# --------------------------------------------------------------------------- #
def test_feedback_fn_is_invoked_online():
    env = TeachingEnvironment(seed=2)
    task = build_teaching_task(repeats=1, seed=2)
    seen = []

    def my_feedback(sample, generation):
        # Custom / label-free: derive a signal ourselves.
        seen.append((sample.id, generation.answer))
        return Feedback(correct=True, signal="env: ok")

    ace = ACE(SimulatedLLM(env), ACEConfig(use_labels=False))
    result = ace.adapt_online(task, feedback_fn=my_feedback)
    assert len(seen) == len(task.samples)
    # Every step graded via our hook → 100% "correct".
    assert result.accuracy == 100.0


def test_feedback_fn_receives_generation():
    env = TeachingEnvironment(seed=2)
    task = build_teaching_task(repeats=1, seed=2)
    got_generation = {}

    def my_feedback(sample, generation):
        got_generation["has"] = isinstance(generation, Generation)
        got_generation["answer"] = generation.answer
        return Feedback(correct=False, ground_truth=sample.answer)

    ace = ACE(SimulatedLLM(env))
    ace.adapt_offline(task, feedback_fn=my_feedback)
    assert got_generation["has"] is True
    assert isinstance(got_generation["answer"], str)


def test_label_free_without_hook_does_not_grade():
    env = TeachingEnvironment(seed=2)
    task = build_teaching_task(repeats=1, seed=2)
    ace = ACE(SimulatedLLM(env), ACEConfig(use_labels=False))
    result = ace.adapt_online(task)  # no labels, no hook
    graded = [r for r in result.history if r.correct is not None]
    assert graded == []  # nothing to learn from → nothing graded


def test_backward_compatible_default_path():
    # Existing labeled behavior unchanged when no feedback_fn is passed.
    env = TeachingEnvironment(seed=2)
    task = build_teaching_task(repeats=3, seed=2)
    train, test = task.split(seed=2)
    ace = ACE(SimulatedLLM(env), ACEConfig(epochs=3))
    ace.adapt_offline(train)
    assert ace.evaluate(test).accuracy > 0


# --------------------------------------------------------------------------- #
# #2  LLM-driven Curator (with deterministic fallback)
# --------------------------------------------------------------------------- #
class CuratorLLM:
    """Fake LLM that emits ADD + UPDATE + REMOVE operations."""

    def __init__(self, target_id):
        self.target_id = target_id
        self.calls = 0

    def complete(self, system, user, **kw):
        return ""

    def complete_json(self, system, user, **kw):
        self.calls += 1
        return {"operations": [
            {"op": "ADD", "section": "strategies", "content": "brand new lesson"},
            {"op": "UPDATE", "target_id": self.target_id, "content": "sharpened"},
            {"op": "REMOVE", "target_id": self.target_id},
        ]}


def test_curator_calls_llm_for_update_and_remove():
    pb = Playbook()
    from ace.playbook import Bullet
    b = pb.add(Bullet(content="old", section="strategies"))
    llm = CuratorLLM(target_id=b.id)
    refl = Reflection(insights=[{"content": "x", "section": "strategies", "tags": []}])
    delta = Curator(llm, use_llm=True).curate(
        Sample(id="1", question="q"), Generation(answer="a"), refl, pb
    )
    assert llm.calls == 1
    kinds = [o.op for o in delta.operations]
    assert DeltaOp.ADD in kinds and DeltaOp.UPDATE in kinds and DeltaOp.REMOVE in kinds


def test_curator_falls_back_when_llm_returns_nothing():
    class EmptyLLM:
        def complete(self, s, u, **k): return ""
        def complete_json(self, s, u, **k): return {"operations": []}

    refl = Reflection(insights=[{"content": "fallback lesson", "section": "domain_concepts", "tags": []}])
    delta = Curator(EmptyLLM(), use_llm=True).curate(
        Sample(id="1", question="q"), Generation(answer="a"), refl, Playbook()
    )
    # Lesson preserved via deterministic ADD fallback.
    assert len(delta.operations) == 1
    assert delta.operations[0].op is DeltaOp.ADD
    assert delta.operations[0].content == "fallback lesson"


def test_curator_falls_back_on_llm_exception():
    class BoomLLM:
        def complete(self, s, u, **k): return ""
        def complete_json(self, s, u, **k): raise RuntimeError("boom")

    refl = Reflection(insights=[{"content": "resilient lesson", "section": "strategies", "tags": []}])
    delta = Curator(BoomLLM(), use_llm=True).curate(
        Sample(id="1", question="q"), Generation(answer="a"), refl, Playbook()
    )
    assert delta.operations and delta.operations[0].content == "resilient lesson"


def test_curator_use_llm_false_stays_deterministic():
    llm = CuratorLLM(target_id="ctx-x")
    refl = Reflection(insights=[{"content": "det", "section": "strategies", "tags": []}])
    delta = Curator(llm, use_llm=False).curate(
        Sample(id="1", question="q"), Generation(answer="a"), refl, Playbook()
    )
    assert llm.calls == 0
    assert all(o.op is DeltaOp.ADD for o in delta.operations)


def test_simulated_backend_never_calls_curator_llm():
    # SimulatedLLM path must remain deterministic regardless of use_llm.
    env = TeachingEnvironment(seed=1)
    ace = ACE(SimulatedLLM(env), ACEConfig(curator_use_llm=True))
    task = build_teaching_task(repeats=1, seed=1)
    result = ace.adapt_online(task)
    assert result.summary()["steps"] == len(task.samples)
