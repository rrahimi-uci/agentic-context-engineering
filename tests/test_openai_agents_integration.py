"""Integration tests for the OpenAI Agents SDK wrapper (``ACEAgent``).

These run against the *real* ``agents`` SDK (so the wrapper stays compatible
with it), but the network-bound ``Runner`` is monkeypatched so no API key or
HTTP call is needed. The whole module is skipped when the SDK is not installed,
so the core test suite still runs with only the base dependencies.
"""

import asyncio

import pytest

agents = pytest.importorskip("agents")

from ace import ACE, SimulatedLLM, TeachingEnvironment
from ace.integrations.openai_agents import ACEAgent, ACERunOutput, playbook_instructions
from ace.playbook import Bullet


class _FakeItem:
    def __init__(self, raw):
        self.raw_item = raw


class _FakeResult:
    """Mimics the shape of an Agents SDK run result."""

    def __init__(self, final_output, items=None):
        self.final_output = final_output
        self.new_items = [_FakeItem(i) for i in (items or [])]


def _ace():
    return ACE(SimulatedLLM(TeachingEnvironment(seed=1)))


def _patch_runner(monkeypatch, result):
    """Patch both the sync and async Runner entry points to return ``result``."""
    monkeypatch.setattr(agents.Runner, "run_sync", staticmethod(lambda agent, q, **kw: result))

    async def _run(agent, q, **kw):
        return result

    monkeypatch.setattr(agents.Runner, "run", staticmethod(_run))


# --------------------------------------------------------------------------- #
# Construction / instruction injection
# --------------------------------------------------------------------------- #
def test_aceagent_wraps_agent_with_dynamic_instructions():
    ace = _ace()
    ace.playbook.add(Bullet(content="always verify identity", section="strategies"))
    base = agents.Agent(name="Support", instructions="You are concise.")
    wrapped = ACEAgent(base, ace=ace)

    # The cloned agent uses a callable (dynamic) instructions hook.
    assert callable(wrapped.agent.instructions)
    text = wrapped.agent.instructions(None, wrapped.agent)
    assert "You are concise." in text
    assert "always verify identity" in text
    assert "Playbook" in text


def test_base_instructions_override():
    ace = _ace()
    base = agents.Agent(name="A", instructions="ORIGINAL")
    wrapped = ACEAgent(base, ace=ace, base_instructions="OVERRIDDEN")
    text = wrapped.agent.instructions(None, None)
    assert "OVERRIDDEN" in text
    assert "ORIGINAL" not in text


def test_callable_base_instructions_are_composed():
    ace = _ace()

    def dyn(ctx, agent):
        return "DYNAMIC BASE"

    base = agents.Agent(name="A", instructions=dyn)
    wrapped = ACEAgent(base, ace=ace)
    text = wrapped.agent.instructions(None, None)
    assert "DYNAMIC BASE" in text


def test_async_base_instructions_fall_back_to_empty():
    ace = _ace()

    async def dyn(ctx, agent):  # cannot be resolved from a sync hook
        return "SHOULD NOT APPEAR"

    fn = playbook_instructions(dyn, ace)
    text = fn(None, None)
    assert "SHOULD NOT APPEAR" not in text
    assert "Playbook" in text  # still produces valid instructions


# --------------------------------------------------------------------------- #
# run / run_and_learn / learn
# --------------------------------------------------------------------------- #
def test_run_returns_output_and_trajectory(monkeypatch):
    _patch_runner(monkeypatch, _FakeResult("hello world", items=["step-1", "step-2"]))
    wrapped = ACEAgent(agents.Agent(name="A", instructions="x"), ace=_ace())
    out = wrapped.run("hi")
    assert isinstance(out, ACERunOutput)
    assert out.output == "hello world"
    assert "step-1" in out.trajectory and "step-2" in out.trajectory


def test_run_and_learn_grows_playbook(monkeypatch):
    _patch_runner(monkeypatch, _FakeResult("done"))
    ace = _ace()
    wrapped = ACEAgent(agents.Agent(name="A", instructions="x"), ace=ace)
    before = len(ace.playbook)
    out = wrapped.run_and_learn(
        "Cancel order #C99",
        signal="Policy: cancellation requires identity verification first.",
    )
    assert out.record is not None
    assert out.record.phase == "agent"
    assert len(ace.playbook) >= before  # a step was taken; lessons may be added


def test_run_and_learn_with_label_grades(monkeypatch):
    _patch_runner(monkeypatch, _FakeResult("B"))
    ace = _ace()
    wrapped = ACEAgent(agents.Agent(name="A", instructions="x"), ace=ace)
    out = wrapped.run_and_learn("q", ground_truth="B")
    assert out.record.correct is True
    out2 = wrapped.run_and_learn("q", ground_truth="not-B")
    assert out2.record.correct is False


def test_learn_without_running(monkeypatch):
    ace = _ace()
    wrapped = ACEAgent(agents.Agent(name="A", instructions="x"), ace=ace)
    rec = wrapped.learn("q", "produced answer", signal="env: ok", correct=True)
    assert rec.phase == "agent"
    assert rec.correct is True


def test_custom_sample_id_is_used(monkeypatch):
    _patch_runner(monkeypatch, _FakeResult("done"))
    wrapped = ACEAgent(agents.Agent(name="A", instructions="x"), ace=_ace())
    out = wrapped.run_and_learn("q", signal="s", sample_id="ticket-42")
    assert out.record.sample_id == "ticket-42"


# --------------------------------------------------------------------------- #
# Async API
# --------------------------------------------------------------------------- #
def test_arun_and_learn(monkeypatch):
    _patch_runner(monkeypatch, _FakeResult("async-out", items=["a"]))
    ace = _ace()
    wrapped = ACEAgent(agents.Agent(name="A", instructions="x"), ace=ace)

    out = asyncio.run(wrapped.arun_and_learn("q", signal="env feedback"))
    assert out.output == "async-out"
    assert out.record is not None and out.record.phase == "agent"


# --------------------------------------------------------------------------- #
# Trajectory extraction edge cases
# --------------------------------------------------------------------------- #
def test_extract_trajectory_falls_back_to_final_output():
    class _Bare:
        final_output = "only final"

    assert ACEAgent._extract_trajectory(_Bare()) == "only final"


def test_extract_trajectory_is_truncated():
    class _Big:
        new_items = [type("I", (), {"raw_item": "x" * 20000})()]

    traj = ACEAgent._extract_trajectory(_Big())
    assert len(traj) <= 8000
