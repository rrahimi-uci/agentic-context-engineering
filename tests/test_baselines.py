"""Tests for the comparison baselines, including context collapse."""

from ace.baselines import MonolithicRewriteAgent, StaticAgent
from ace.llm import SimulatedLLM
from ace.tasks import TeachingEnvironment, build_teaching_task


def test_static_agent_has_no_playbook_and_grades():
    env = TeachingEnvironment(seed=1)
    task = build_teaching_task(repeats=1, seed=1)
    result = StaticAgent(SimulatedLLM(env)).run(task)
    assert len(result.history) == len(task.samples)
    assert all(r.playbook_size == 0 for r in result.history)
    # Base model is graded against labels.
    assert all(r.correct is not None for r in result.history)


def test_monolithic_rewrite_collapses_deterministically():
    env = TeachingEnvironment(seed=1)
    task = build_teaching_task(repeats=3, seed=1)
    agent = MonolithicRewriteAgent(SimulatedLLM(env))
    result = agent.run(task)
    collapses = [r for r in result.history if r.refine.get("collapsed")]
    assert collapses, "expected at least one context collapse over a long run"
    # Collapse is reproducible for the same step index.
    assert agent._collapses(1) == agent._collapses(1)


def test_monolithic_is_reproducible_across_runs():
    task = build_teaching_task(repeats=3, seed=1)
    a = MonolithicRewriteAgent(SimulatedLLM(TeachingEnvironment(seed=1))).run(task)
    b = MonolithicRewriteAgent(SimulatedLLM(TeachingEnvironment(seed=1))).run(task)
    assert a.accuracy == b.accuracy
    assert [r.refine.get("collapsed") for r in a.history] == [
        r.refine.get("collapsed") for r in b.history
    ]


def test_monolithic_learns_rules_into_playbook():
    env = TeachingEnvironment(seed=2)
    task = build_teaching_task(repeats=2, seed=2)
    result = MonolithicRewriteAgent(SimulatedLLM(env), collapse_prob=0.0).run(task)
    # With no collapses, accumulated knowledge should produce a non-empty playbook.
    assert result.playbook is not None and len(result.playbook) > 0
