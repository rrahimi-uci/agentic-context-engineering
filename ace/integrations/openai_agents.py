"""First-class integration with the **OpenAI Agents SDK**.

This makes ACE a drop-in *self-improving memory* for an
``agents.Agent``. The agent's instructions are augmented at run time with the
ACE playbook (rendered context), the agent's real execution trajectory is
captured, and the Reflector/Curator turn that trajectory into incremental
playbook updates.

Minimal usage
-------------
```python
from agents import Agent, function_tool
from ace import ACE, OpenAILLM
from ace.integrations.openai_agents import ACEAgent

base = Agent(name="Support", instructions="You are a helpful support agent.")
llm = OpenAILLM(model="gpt-4o-mini")
agent = ACEAgent(base, ace=ACE(llm))

# Run + learn from natural execution feedback (no labels needed):
out = agent.run_and_learn("Refund order #123", signal="tool refund() returned 200 OK")
print(out.output)
print(agent.ace.playbook.render())
```

Both synchronous (:meth:`ACEAgent.run`, :meth:`ACEAgent.run_and_learn`) and
asynchronous (:meth:`ACEAgent.arun`, :meth:`ACEAgent.arun_and_learn`) entry
points are provided. The async variants are the right choice inside an existing
event loop (FastAPI, notebooks, other async agents) where the SDK's
``run_sync`` cannot be used.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

from ..engine import ACE, StepRecord
from ..feedback import Feedback
from ..roles import Generation
from ..tasks import Sample

# Base instructions may be a plain string or the SDK's "dynamic instructions"
# callable ``(run_context, agent) -> str``.
BaseInstructions = Union[str, Callable[[Any, Any], Any]]


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
                base.close()  # type: ignore[union-attr]
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


@dataclass
class ACERunOutput:
    output: str
    trajectory: str
    record: Optional[StepRecord] = None
    raw_result: Any = None


class ACEAgent:
    """Wraps an ``agents.Agent`` with an ACE playbook that learns over time.

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
    """

    def __init__(
        self,
        base_agent: Any,
        ace: ACE,
        *,
        base_instructions: Optional[BaseInstructions] = None,
    ) -> None:
        try:
            from agents import Agent  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ACEAgent requires the OpenAI Agents SDK. Install with "
                "`pip install ace-playbook[agents]`."
            ) from exc

        self.ace = ace
        resolved = base_instructions if base_instructions is not None else base_agent.instructions
        # A plain (non-callable) base that is not a string is coerced to "".
        if not callable(resolved) and not isinstance(resolved, str):
            resolved = ""
        self._base_instructions: BaseInstructions = resolved or ""
        # Clone the agent with dynamic, playbook-augmented instructions.
        self.agent = base_agent.clone(
            instructions=playbook_instructions(self._base_instructions, ace)
        )

    # ------------------------------------------------------------------ #
    # Synchronous API
    # ------------------------------------------------------------------ #
    def run(self, query: str, **runner_kwargs) -> ACERunOutput:
        """Run the agent once (no learning)."""
        from agents import Runner

        result = Runner.run_sync(self.agent, query, **runner_kwargs)
        return self._to_output(result)

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
        or ``signal`` (natural execution feedback — the label-free path).
        """
        run_out = self.run(query, **runner_kwargs)
        run_out.record = self._learn(
            query, run_out, ground_truth=ground_truth, correct=correct,
            signal=signal, sample_id=sample_id,
        )
        return run_out

    # ------------------------------------------------------------------ #
    # Asynchronous API (for use inside an existing event loop)
    # ------------------------------------------------------------------ #
    async def arun(self, query: str, **runner_kwargs) -> ACERunOutput:
        """Run the agent once (no learning), asynchronously."""
        from agents import Runner

        result = await Runner.run(self.agent, query, **runner_kwargs)
        return self._to_output(result)

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
        run_out.record = self._learn(
            query, run_out, ground_truth=ground_truth, correct=correct,
            signal=signal, sample_id=sample_id,
        )
        return run_out

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
            query, run_out, ground_truth=ground_truth, correct=correct,
            signal=signal, sample_id=sample_id,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
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
        return self.ace.step(sample, feedback, phase="agent", generation=gen)

    def _to_output(self, result: Any) -> ACERunOutput:
        output = str(getattr(result, "final_output", ""))
        return ACERunOutput(
            output=output, trajectory=self._extract_trajectory(result), raw_result=result
        )

    @staticmethod
    def _extract_trajectory(result: Any) -> str:
        """Best-effort flatten of the agent run into a trajectory string."""
        parts: List[str] = []
        for attr in ("new_items", "items"):
            items = getattr(result, attr, None)
            if items:
                for it in items:
                    parts.append(str(getattr(it, "raw_item", it)))
                break
        if not parts:
            parts.append(str(getattr(result, "final_output", "")))
        return "\n".join(parts)[:8000]
