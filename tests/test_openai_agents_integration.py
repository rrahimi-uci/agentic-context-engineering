"""Integration tests for the OpenAI Agents SDK wrapper (``ACEAgent``).

These run against the *real* ``agents`` SDK (so the wrapper stays compatible
with it), but the network-bound ``Runner`` is monkeypatched so no API key or
HTTP call is needed. The whole module is skipped when the SDK is not installed,
so the core test suite still runs with only the base dependencies.
"""

import asyncio
import os

import pytest

agents = pytest.importorskip("agents")
# The wrapper opens an ``ace.learn`` trace on every learn; disable export so the
# test suite never touches the network.
agents.set_tracing_disabled(True)

import ace as ace_pkg
from ace import ACE, SimulatedLLM, TeachingEnvironment, wrap_agent
from ace.integrations.openai_agents import (
    ACEAgent,
    ACERunOutput,
    _final_output_text,
    _run_hooks_class,
    playbook_instructions,
)
from ace.playbook import Bullet, Playbook


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


# --------------------------------------------------------------------------- #
# Helpers shared by the new tests
# --------------------------------------------------------------------------- #
class _FakeStream:
    """Mimics the shape of an Agents SDK ``RunResultStreaming``."""

    def __init__(self, final_output, items=None):
        self.final_output = final_output
        self.new_items = [_FakeItem(i) for i in (items or [])]

    async def stream_events(self):
        for e in ("e1", "e2"):
            yield e


def _capture_run_sync(monkeypatch, result=None, store=None):
    """Patch ``run_sync`` to record the kwargs it was called with."""
    result = result if result is not None else _FakeResult("ok")
    store = store if store is not None else {}

    def _sync(agent, q, **kw):
        store.update(kw)
        return result

    monkeypatch.setattr(agents.Runner, "run_sync", staticmethod(_sync))
    return store


# --------------------------------------------------------------------------- #
# Top-level lazy re-exports (crystal-clear API)
# --------------------------------------------------------------------------- #
def test_top_level_reexports_resolve():
    assert ace_pkg.ACEAgent is ACEAgent
    assert ace_pkg.wrap_agent is wrap_agent
    assert ace_pkg.ACERunOutput is ACERunOutput
    assert "wrap_agent" in dir(ace_pkg)


def test_unknown_top_level_attr_raises():
    with pytest.raises(AttributeError):
        ace_pkg.does_not_exist  # noqa: B018


# --------------------------------------------------------------------------- #
# wrap_agent factory
# --------------------------------------------------------------------------- #
def test_wrap_agent_with_explicit_ace_injects_playbook():
    ace = _ace()
    ace.playbook.add(Bullet(content="verify identity", section="strategies"))
    agent = wrap_agent(agents.Agent(name="S", instructions="be nice"), ace=ace)
    assert isinstance(agent, ACEAgent)
    text = agent.agent.instructions(None, agent.agent)
    assert "be nice" in text and "verify identity" in text


def test_wrap_agent_accepts_playbook_object():
    pb = Playbook()
    pb.add(Bullet(content="rule X", section="strategies"))
    agent = wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace(), playbook=pb)
    assert agent.playbook is pb


# --------------------------------------------------------------------------- #
# Playbook persistence
# --------------------------------------------------------------------------- #
def test_playbook_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "mem.json")
    ace1 = _ace()
    a1 = wrap_agent(agents.Agent(name="S", instructions="x"), ace=ace1, playbook=path)
    ace1.playbook.add(Bullet(content="always verify identity", section="strategies"))
    assert a1.save() == path and os.path.exists(path)

    n = len(a1.playbook)
    assert n >= 1
    # A fresh agent constructed on the same path loads the persisted playbook.
    a2 = wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace(), playbook=path)
    assert len(a2.playbook) == n


def test_save_without_path_raises():
    a = wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace())
    with pytest.raises(ValueError):
        a.save()


# --------------------------------------------------------------------------- #
# Capture hooks (auto-learn from tool errors)
# --------------------------------------------------------------------------- #
def test_capture_hooks_attached_by_default(monkeypatch):
    captured = _capture_run_sync(monkeypatch)
    wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace()).run("hi")
    # ``agents.RunHooks`` is a subscripted generic alias (no isinstance), so we
    # check against the concrete listener class the wrapper builds.
    assert isinstance(captured.get("hooks"), _run_hooks_class())


def test_user_hooks_are_not_overridden(monkeypatch):
    captured = _capture_run_sync(monkeypatch)
    sentinel = agents.RunHooks()
    wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace()).run("hi", hooks=sentinel)
    assert captured["hooks"] is sentinel


def test_capture_can_be_disabled(monkeypatch):
    captured = _capture_run_sync(monkeypatch)
    wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace(), capture=False).run("hi")
    assert "hooks" not in captured


def test_ace_run_hooks_capture_trajectory_and_signal():
    hooks_cls = _run_hooks_class()
    assert hooks_cls is not None

    class _Tool:
        name = "refund"

    h = hooks_cls()
    asyncio.run(h.on_tool_start(None, None, _Tool()))
    asyncio.run(h.on_tool_end(None, None, _Tool(), "Error: API returned 500"))
    assert "refund" in h.trajectory()
    assert "refund" in h.signal()

    clean = hooks_cls()
    asyncio.run(clean.on_tool_end(None, None, _Tool(), "200 OK, refund complete"))
    assert clean.signal() == ""


def test_run_and_learn_auto_signal_from_tool_error(monkeypatch):
    async def _fire(h):
        class _Tool:
            name = "refund"

        await h.on_tool_start(None, None, _Tool())
        await h.on_tool_end(None, None, _Tool(), "Error: refund API returned 500")

    def _sync(agent, q, **kw):
        h = kw.get("hooks")
        if h is not None:
            asyncio.run(_fire(h))
        return _FakeResult("could not refund")

    monkeypatch.setattr(agents.Runner, "run_sync", staticmethod(_sync))
    agent = wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace())
    out = agent.run_and_learn("Refund #1")  # no explicit feedback given
    assert "refund" in out.auto_signal.lower()
    assert "[tool_output]" in out.trajectory
    assert out.record is not None and out.record.phase == "agent"


# --------------------------------------------------------------------------- #
# Session passthrough & structured output
# --------------------------------------------------------------------------- #
def test_session_is_forwarded_to_runner(monkeypatch):
    captured = _capture_run_sync(monkeypatch)
    sess = object()
    wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace()).run("hi", session=sess)
    assert captured.get("session") is sess


def test_structured_final_output_is_serialized():
    class _Model:
        def model_dump_json(self):
            return '{"a": 1}'

    assert _final_output_text(type("R", (), {"final_output": _Model()})()) == '{"a": 1}'
    assert _final_output_text(type("R", (), {"final_output": None})()) == ""


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def test_arun_streamed_and_learn(monkeypatch):
    monkeypatch.setattr(
        agents.Runner,
        "run_streamed",
        staticmethod(lambda agent, q, **kw: _FakeStream("streamed-out", items=["s1"])),
    )
    events = []
    agent = wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace())
    out = asyncio.run(
        agent.arun_streamed_and_learn("q", signal="env feedback", on_event=events.append)
    )
    assert out.output == "streamed-out"
    assert events == ["e1", "e2"]
    assert out.record is not None and out.record.phase == "agent"


def test_stream_returns_raw_streaming_result(monkeypatch):
    sentinel = _FakeStream("x")
    monkeypatch.setattr(
        agents.Runner, "run_streamed", staticmethod(lambda agent, q, **kw: sentinel)
    )
    agent = wrap_agent(agents.Agent(name="S", instructions="x"), ace=_ace())
    assert agent.stream("q") is sentinel
