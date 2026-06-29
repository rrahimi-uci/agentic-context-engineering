"""First-class integration with the **OpenAI Agents SDK**.

This makes ACE a drop-in *self-improving memory* for an ``agents.Agent``. The
agent's instructions are augmented at run time with the ACE playbook (rendered
context), the agent's real execution trajectory is captured, and the
Reflector/Curator turn that trajectory into incremental playbook updates.

Crystal-clear usage
-------------------
```python
from agents import Agent
from ace import wrap_agent          # one import, top-level

agent = wrap_agent(
    Agent(name="Support", instructions="You are a helpful support agent."),
    model="gpt-4o-mini",
    playbook="support_memory.json",  # load if it exists; persists what it learns
)

# Run + learn from natural execution feedback (no labels needed):
out = agent.run_and_learn("Refund order #123", signal="tool refund() returned 200 OK")
print(out.output)
agent.save()                         # the agent's learned rules survive a restart
```

What you get
------------
* **Playbook injection** — the current playbook is appended to the agent's
  instructions on every run (string *or* dynamic-callable base instructions are
  supported and composed).
* **Rich trajectory capture** — tool calls, tool outputs, and messages are
  rendered with the SDK's :class:`ItemHelpers` / typed run-items, so the
  Reflector learns from *what actually happened*, not a raw repr.
* **Auto-learn from tool errors** — when ``capture=True`` (default), a
  :class:`RunHooks` listener records the run; if a tool errored and you gave no
  explicit feedback, that error becomes the execution signal automatically.
* **Tracing** — the reflect/curate/merge step is emitted as an ``ace.learn``
  span so playbook growth shows up in the OpenAI trace UI next to the run.
* **Sessions** — multi-turn conversation memory is orthogonal to ACE's learned
  memory; pass the SDK's ``session=`` straight through any ``run`` method.

Entry points
------------
* Sync: :meth:`ACEAgent.run`, :meth:`ACEAgent.run_and_learn`
* Async: :meth:`ACEAgent.arun`, :meth:`ACEAgent.arun_and_learn`
* Streaming: :meth:`ACEAgent.stream` (raw control),
  :meth:`ACEAgent.arun_streamed_and_learn` (batteries-included)
* Learn from an already-produced answer: :meth:`ACEAgent.learn`

The async variants are the right choice inside an existing event loop (FastAPI,
notebooks, other async agents) where the SDK's ``run_sync`` cannot be used.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Union

from ..engine import ACE, StepRecord
from ..feedback import Feedback
from ..playbook import Playbook
from ..roles import Generation
from ..tasks import Sample

# Base instructions may be a plain string or the SDK's "dynamic instructions"
# callable ``(run_context, agent) -> str``.
BaseInstructions = Union[str, Callable[[Any, Any], Any]]

# Trajectories are capped so a runaway tool log can't blow up a Reflector prompt.
MAX_TRAJ_CHARS = 8000

_ERROR_RE = re.compile(
    r"\b(error|errors|exception|traceback|failed|failure|denied|invalid|"
    r"timeout|timed out|not found|unauthorized|forbidden|4\d\d|5\d\d)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Instruction injection
# --------------------------------------------------------------------------- #
def playbook_instructions(base_instructions: BaseInstructions, ace: ACE):
    """Build a *dynamic instructions* callable for an ``agents.Agent``.

    The returned function is called by the Agents SDK on every run and injects
    the current ACE playbook beneath the agent's base instructions. The base may
    itself be the SDK's dynamic-instructions callable; if it is, we resolve it
    first and compose. (An *async* base-instructions callable cannot be resolved
    from this synchronous hook and is treated as empty — pass a resolved string
    via ``ACEAgent(base, ace, base_instructions=...)`` in that case.)
    """

    def _instructions(run_context: Any, agent: Any) -> str:
        base = base_instructions
        if callable(base):
            try:
                base = base(run_context, agent)
            except Exception:
                base = ""
        if inspect.isawaitable(base):  # async dynamic instructions — can't await here
            try:
                base.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            base = ""
        rendered = ace.playbook.render()
        return (
            f"{base or ''}\n\n"
            "# Playbook (accumulated strategies, domain knowledge, and pitfalls)\n"
            "Use the following playbook. Prefer its strategies; avoid its known mistakes.\n\n"
            f"{rendered}"
        )

    return _instructions


# --------------------------------------------------------------------------- #
# Trajectory rendering (uses the SDK's typed run-items when available)
# --------------------------------------------------------------------------- #
def _short(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"


def _looks_like_error(text: str) -> bool:
    return bool(_ERROR_RE.search(text or ""))


def _agents_item_api():
    """Lazily import the SDK's run-item types; ``None`` if the SDK is absent."""
    try:
        from agents import (
            ItemHelpers,
            MessageOutputItem,
            ReasoningItem,
            ToolCallItem,
            ToolCallOutputItem,
        )
    except Exception:
        return None
    return {
        "ItemHelpers": ItemHelpers,
        "MessageOutputItem": MessageOutputItem,
        "ReasoningItem": ReasoningItem,
        "ToolCallItem": ToolCallItem,
        "ToolCallOutputItem": ToolCallOutputItem,
    }


def _describe_tool_call(raw: Any) -> str:
    name = getattr(raw, "name", None) or "tool"
    args = getattr(raw, "arguments", None)
    if args is None:
        return f"{name}()"
    return f"{name}({_short(str(args), 500)})"


def _render_one_item(it: Any, api: Optional[dict]) -> str:
    """Render a single run item into one readable trajectory line."""
    if api is not None:
        if isinstance(it, api["MessageOutputItem"]):
            try:
                return f"[assistant] {api['ItemHelpers'].text_message_output(it)}"
            except Exception:
                pass
        elif isinstance(it, api["ToolCallItem"]):
            return f"[tool_call] {_describe_tool_call(getattr(it, 'raw_item', None))}"
        elif isinstance(it, api["ToolCallOutputItem"]):
            return f"[tool_output] {_short(str(getattr(it, 'output', '')), 1000)}"
        elif isinstance(it, api["ReasoningItem"]):
            return ""  # internal reasoning — skip from the learned trajectory
    # Generic fallback for unknown / non-SDK items (and the test fakes).
    return str(getattr(it, "raw_item", it))


def _render_run_items(items: Any) -> List[str]:
    api = _agents_item_api()
    return [line for line in (_render_one_item(it, api) for it in items) if line]


# --------------------------------------------------------------------------- #
# Auto-learn RunHooks (built lazily so importing this module never needs the SDK)
# --------------------------------------------------------------------------- #
_RUN_HOOKS_CLS = None


def _run_hooks_class():
    """Return (and cache) an ``ACERunHooks`` class, or ``None`` without the SDK."""
    global _RUN_HOOKS_CLS
    if _RUN_HOOKS_CLS is not None:
        return _RUN_HOOKS_CLS
    try:
        from agents import RunHooks
    except Exception:
        return None

    class ACERunHooks(RunHooks):
        """Records a readable trajectory and surfaces tool errors as a signal."""

        def __init__(self) -> None:
            self.events: List[str] = []
            self.errors: List[str] = []

        async def on_tool_start(self, context, agent, tool) -> None:  # noqa: ANN001
            self.events.append(f"[tool_call] {getattr(tool, 'name', 'tool')}()")

        async def on_tool_end(self, context, agent, tool, result) -> None:  # noqa: ANN001
            name = getattr(tool, "name", "tool")
            text = _short(str(result), 1000)
            self.events.append(f"[tool_output] {name} -> {text}")
            if _looks_like_error(text):
                self.errors.append(f"{name}: {text}")

        async def on_handoff(self, context, from_agent, to_agent) -> None:  # noqa: ANN001
            self.events.append(
                f"[handoff] {getattr(from_agent, 'name', '?')} -> {getattr(to_agent, 'name', '?')}"
            )

        def trajectory(self) -> str:
            return "\n".join(self.events)

        def signal(self) -> str:
            if not self.errors:
                return ""
            return "Tool execution issues: " + "; ".join(self.errors[:5])

    _RUN_HOOKS_CLS = ACERunHooks
    return _RUN_HOOKS_CLS


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
def _agents_trace_api():
    try:
        from agents import custom_span, get_current_trace, trace
    except Exception:
        return None
    return trace, custom_span, get_current_trace


# --------------------------------------------------------------------------- #
# Final-output extraction (handles structured ``output_type`` results)
# --------------------------------------------------------------------------- #
def _final_output_text(result: Any) -> str:
    fo = getattr(result, "final_output", "")
    if fo is None:
        return ""
    if isinstance(fo, str):
        return fo
    dump = getattr(fo, "model_dump_json", None)  # pydantic output_type
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return str(fo)


@dataclass
class ACERunOutput:
    """The result of a wrapped run.

    ``auto_signal`` is the execution signal derived from captured tool errors
    (empty when nothing went wrong); ``events`` is the captured trajectory log.
    """

    output: str
    trajectory: str
    record: Optional[StepRecord] = None
    raw_result: Any = None
    auto_signal: str = ""
    events: List[str] = field(default_factory=list)


class ACEAgent:
    """Wraps an ``agents.Agent`` with an ACE playbook that learns over time.

    Most users should reach for :func:`wrap_agent` instead — it builds the ACE
    engine for you. Use ``ACEAgent`` directly when you want to share a
    pre-configured :class:`~ace.engine.ACE` across several agents.

    Parameters
    ----------
    base_agent:
        An ``agents.Agent``. Its ``instructions`` are used as the base (a string,
        or the SDK's dynamic-instructions callable); the playbook is appended
        dynamically on each run.
    ace:
        The :class:`~ace.engine.ACE` engine whose Reflector/Curator update the
        playbook. Use an :class:`~ace.llm.OpenAILLM` backend in production.
    base_instructions:
        Optional explicit base instructions, overriding ``base_agent.instructions``.
        Useful when the base agent uses *async* dynamic instructions (which this
        wrapper cannot resolve on its own).
    playbook_path:
        If given, the playbook is loaded from this path at construction (when the
        file exists) and becomes the default target for :meth:`save`.
    capture:
        When True (default), a :class:`RunHooks` listener records each run for a
        precise tool-level trajectory and turns tool errors into an automatic
        execution signal. Disabled automatically if you pass your own ``hooks=``.
    trace:
        When True (default), the learning step is emitted as an ``ace.learn``
        tracing span.
    """

    def __init__(
        self,
        base_agent: Any,
        ace: ACE,
        *,
        base_instructions: Optional[BaseInstructions] = None,
        playbook_path: Optional[str] = None,
        capture: bool = True,
        trace: bool = True,
    ) -> None:
        try:
            from agents import Agent  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ACEAgent requires the OpenAI Agents SDK. Install with "
                "`pip install ace-playbook[agents]`."
            ) from exc

        self.ace = ace
        self._capture = capture
        self._trace = trace
        self.playbook_path = str(playbook_path) if playbook_path else None
        # Load a persisted playbook into the engine if one exists at the path.
        if self.playbook_path and os.path.exists(self.playbook_path):
            self.ace.playbook = Playbook.load(self.playbook_path)

        resolved = base_instructions if base_instructions is not None else base_agent.instructions
        # A plain (non-callable) base that is not a string is coerced to "".
        if not callable(resolved) and not isinstance(resolved, str):
            resolved = ""
        self._base_instructions: BaseInstructions = resolved or ""
        # Clone the agent with dynamic, playbook-augmented instructions. The
        # closure reads ``ace.playbook`` lazily, so replacing it above is fine.
        self.agent = base_agent.clone(
            instructions=playbook_instructions(self._base_instructions, ace)
        )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def playbook(self) -> Playbook:
        """The live playbook the agent is learning into."""
        return self.ace.playbook

    def save(self, path: Optional[str] = None) -> str:
        """Persist the learned playbook to ``path`` (or the construction path)."""
        target = path or self.playbook_path
        if not target:
            raise ValueError(
                "No playbook path set. Call save('memory.json') or construct with "
                "wrap_agent(..., playbook='memory.json')."
            )
        self.ace.playbook.save(target)
        self.playbook_path = target
        return target

    # ------------------------------------------------------------------ #
    # Synchronous API
    # ------------------------------------------------------------------ #
    def run(self, query: str, **runner_kwargs) -> ACERunOutput:
        """Run the agent once (no learning). Forwards ``session=``, ``context=``,
        ``max_turns=``, ``hooks=``, ``run_config=`` straight to the SDK Runner."""
        from agents import Runner

        hooks, rk = self._maybe_attach_hooks(runner_kwargs)
        result = Runner.run_sync(self.agent, query, **rk)
        return self._to_output(result, hooks)

    def run_and_learn(
        self,
        query: str,
        *,
        ground_truth: Optional[str] = None,
        correct: Optional[bool] = None,
        signal: str = "",
        sample_id: Optional[str] = None,
        **runner_kwargs,
    ) -> ACERunOutput:
        """Run the agent, then update the playbook from the trajectory + feedback.

        Provide any of ``ground_truth`` (labeled), ``correct`` (a boolean reward),
        or ``signal`` (natural execution feedback — the label-free path). If you
        provide none and ``capture`` is on, a tool error captured during the run
        becomes the signal automatically.
        """
        run_out = self.run(query, **runner_kwargs)
        eff_signal = self._effective_signal(signal, run_out, ground_truth, correct)
        run_out.record = self._learn(
            query,
            run_out,
            ground_truth=ground_truth,
            correct=correct,
            signal=eff_signal,
            sample_id=sample_id,
        )
        return run_out

    # ------------------------------------------------------------------ #
    # Asynchronous API (for use inside an existing event loop)
    # ------------------------------------------------------------------ #
    async def arun(self, query: str, **runner_kwargs) -> ACERunOutput:
        """Run the agent once (no learning), asynchronously."""
        from agents import Runner

        hooks, rk = self._maybe_attach_hooks(runner_kwargs)
        result = await Runner.run(self.agent, query, **rk)
        return self._to_output(result, hooks)

    async def arun_and_learn(
        self,
        query: str,
        *,
        ground_truth: Optional[str] = None,
        correct: Optional[bool] = None,
        signal: str = "",
        sample_id: Optional[str] = None,
        **runner_kwargs,
    ) -> ACERunOutput:
        """Async counterpart of :meth:`run_and_learn`."""
        run_out = await self.arun(query, **runner_kwargs)
        eff_signal = self._effective_signal(signal, run_out, ground_truth, correct)
        run_out.record = await self._alearn(
            query,
            run_out,
            ground_truth=ground_truth,
            correct=correct,
            signal=eff_signal,
            sample_id=sample_id,
        )
        return run_out

    # ------------------------------------------------------------------ #
    # Streaming API
    # ------------------------------------------------------------------ #
    def stream(self, query: str, **runner_kwargs):
        """Start a streamed run and return the SDK's ``RunResultStreaming``.

        Full-control escape hatch: iterate ``result.stream_events()`` yourself,
        then call :meth:`learn` with the final output. For the batteries-included
        path use :meth:`arun_streamed_and_learn`.
        """
        from agents import Runner

        _, rk = self._maybe_attach_hooks(runner_kwargs)
        return Runner.run_streamed(self.agent, query, **rk)

    async def arun_streamed_and_learn(
        self,
        query: str,
        *,
        ground_truth: Optional[str] = None,
        correct: Optional[bool] = None,
        signal: str = "",
        sample_id: Optional[str] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        **runner_kwargs,
    ) -> ACERunOutput:
        """Stream the run to completion (optionally calling ``on_event`` per event),
        then learn from the finished trajectory."""
        from agents import Runner

        hooks, rk = self._maybe_attach_hooks(runner_kwargs)
        streamed = Runner.run_streamed(self.agent, query, **rk)
        async for event in streamed.stream_events():
            if on_event is not None:
                on_event(event)
        run_out = self._to_output(streamed, hooks)
        eff_signal = self._effective_signal(signal, run_out, ground_truth, correct)
        run_out.record = await self._alearn(
            query,
            run_out,
            ground_truth=ground_truth,
            correct=correct,
            signal=eff_signal,
            sample_id=sample_id,
        )
        return run_out

    # ------------------------------------------------------------------ #
    # Learn from an already-produced answer
    # ------------------------------------------------------------------ #
    def learn(
        self,
        query: str,
        output: str,
        *,
        ground_truth: Optional[str] = None,
        correct: Optional[bool] = None,
        signal: str = "",
        trajectory: str = "",
        sample_id: Optional[str] = None,
    ) -> StepRecord:
        """Update the playbook from an already-produced (query, output) pair."""
        run_out = ACERunOutput(output=output, trajectory=trajectory)
        return self._learn(
            query,
            run_out,
            ground_truth=ground_truth,
            correct=correct,
            signal=signal,
            sample_id=sample_id,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _maybe_attach_hooks(self, runner_kwargs: dict):
        """Attach an auto-learn RunHooks listener unless disabled / user-supplied."""
        if not self._capture or "hooks" in runner_kwargs:
            return None, runner_kwargs
        cls = _run_hooks_class()
        if cls is None:
            return None, runner_kwargs
        hooks = cls()
        rk = dict(runner_kwargs)
        rk["hooks"] = hooks
        return hooks, rk

    @staticmethod
    def _effective_signal(
        signal: str, run_out: ACERunOutput, ground_truth: Optional[str], correct: Optional[bool]
    ) -> str:
        if signal:
            return signal
        # Only auto-derive a signal when the caller gave no other feedback.
        if ground_truth is None and correct is None:
            return run_out.auto_signal or ""
        return ""

    def _learn(
        self,
        query: str,
        run_out: ACERunOutput,
        *,
        ground_truth: Optional[str],
        correct: Optional[bool],
        signal: str,
        sample_id: Optional[str],
    ) -> StepRecord:
        sample = Sample(
            id=sample_id or f"q-{self.ace._step + 1}",
            question=query,
            answer=ground_truth or "",
        )
        gen = Generation(answer=run_out.output, reasoning=run_out.trajectory)
        feedback = Feedback(correct=correct, ground_truth=ground_truth, signal=signal)
        with self._learn_trace():
            return self.ace.step(sample, feedback, phase="agent", generation=gen)

    async def _alearn(self, query: str, run_out: ACERunOutput, **feedback) -> StepRecord:
        """Run the synchronous learn step off the event loop.

        ACE's Reflector/Curator make blocking LLM calls; offloading to a worker
        thread keeps the caller's event loop (FastAPI, notebooks, other async
        agents) responsive. ``asyncio.to_thread`` copies the current context, so
        the ``ace.learn`` tracing span still nests correctly.
        """
        return await asyncio.to_thread(self._learn, query, run_out, **feedback)

    @contextmanager
    def _learn_trace(self) -> Iterator[None]:
        """Emit an ``ace.learn`` span; never let tracing break learning."""
        cm: Optional[ExitStack] = None
        if self._trace:
            api = _agents_trace_api()
            if api is not None:
                trace_fn, custom_span_fn, get_current = api
                try:
                    stack = ExitStack()
                    if get_current() is None:
                        stack.enter_context(trace_fn(workflow_name="ACE learning"))
                    stack.enter_context(custom_span_fn("ace.learn"))
                    cm = stack
                except Exception:
                    cm = None
        if cm is None:
            yield
            return
        with cm:
            yield

    def _to_output(self, result: Any, hooks: Any = None) -> ACERunOutput:
        output = _final_output_text(result)
        trajectory = hooks.trajectory() if hooks is not None else ""
        if not trajectory:
            trajectory = self._extract_trajectory(result)
        out = ACERunOutput(output=output, trajectory=trajectory[:MAX_TRAJ_CHARS], raw_result=result)
        if hooks is not None:
            out.auto_signal = hooks.signal()
            out.events = list(hooks.events)
        return out

    @staticmethod
    def _extract_trajectory(result: Any) -> str:
        """Render an agent run into a readable trajectory string.

        Uses the SDK's typed run-items (tool calls, tool outputs, messages) when
        available and falls back to the final output otherwise.
        """
        items = getattr(result, "new_items", None) or getattr(result, "items", None)
        parts: List[str] = _render_run_items(items) if items else []
        if not parts:
            fo = getattr(result, "final_output", "")
            parts = [str(fo)] if fo not in (None, "") else []
        return "\n".join(parts)[:MAX_TRAJ_CHARS]


# --------------------------------------------------------------------------- #
# The crystal-clear entry point
# --------------------------------------------------------------------------- #
def wrap_agent(
    base_agent: Any,
    *,
    model: str = "gpt-4o-mini",
    ace: Optional[ACE] = None,
    config: Optional[Any] = None,
    playbook: Optional[Union[str, "os.PathLike[str]", Playbook]] = None,
    base_instructions: Optional[BaseInstructions] = None,
    capture: bool = True,
    trace: bool = True,
    **llm_kwargs,
) -> ACEAgent:
    """Wrap an ``agents.Agent`` so it learns a playbook from experience.

    This is the one-call path: it builds the ACE engine (an
    :class:`~ace.llm.OpenAILLM` backend on ``model``) for you, optionally loads a
    persisted playbook, and returns a ready :class:`ACEAgent`.

    Parameters
    ----------
    base_agent:
        The ``agents.Agent`` to wrap.
    model:
        Chat model id for ACE's Reflector/Curator (ignored if ``ace`` is given).
    ace:
        Supply your own pre-built :class:`~ace.engine.ACE` to skip engine
        construction (e.g. to share one engine across agents, or use a non-OpenAI
        backend). ``model`` / ``llm_kwargs`` are then ignored.
    config:
        Optional :class:`~ace.config.ACEConfig` for the auto-built engine.
    playbook:
        A path (loaded if it exists, and the default :meth:`ACEAgent.save` target)
        or an existing :class:`~ace.playbook.Playbook` to continue.
    base_instructions, capture, trace:
        Passed through to :class:`ACEAgent`.
    **llm_kwargs:
        Extra args for the auto-built ``OpenAILLM`` (e.g. ``api_key``,
        ``temperature``, ``base_url``).

    Examples
    --------
    >>> from agents import Agent
    >>> from ace import wrap_agent
    >>> agent = wrap_agent(Agent(name="Support", instructions="Be concise."),
    ...                    model="gpt-4o-mini", playbook="support.json")
    >>> out = agent.run_and_learn("Cancel #C99", signal="verify identity first")
    >>> agent.save()
    """
    if ace is None:
        from ..engine import ACE as _ACE
        from ..llm import OpenAILLM

        ace = _ACE(OpenAILLM(model=model, **llm_kwargs), config=config)

    playbook_path: Optional[str] = None
    if isinstance(playbook, Playbook):
        ace.playbook = playbook
    elif playbook is not None:
        playbook_path = str(playbook)

    return ACEAgent(
        base_agent,
        ace=ace,
        base_instructions=base_instructions,
        playbook_path=playbook_path,
        capture=capture,
        trace=trace,
    )
