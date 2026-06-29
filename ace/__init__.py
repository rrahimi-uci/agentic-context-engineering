"""ACE — Agentic Context Engineering.

A faithful, framework-style implementation of *"Agentic Context Engineering:
Evolving Contexts for Self-Improving Language Models"* (ICLR 2026).

ACE treats an LLM's context as an evolving **playbook** that accumulates,
refines, and organizes strategies over time through a modular loop of
**Generator → Reflector → Curator**, with incremental **delta updates** and a
**grow-and-refine** mechanism that together prevent *context collapse* and
*brevity bias*.

Quick start
-----------
```python
from ace import ACE, SimulatedLLM, TeachingEnvironment, build_teaching_task

env = TeachingEnvironment()
llm = SimulatedLLM(env)
task = build_teaching_task()
train, test = task.split()

ace = ACE(llm)
ace.adapt_offline(train)          # build a playbook from feedback
result = ace.evaluate(test)       # measure on held-out data
print(result.summary())
print(ace.playbook.render())
```

OpenAI Agents SDK
-----------------
```python
from agents import Agent
from ace import wrap_agent                  # lazily pulls in the integration

agent = wrap_agent(Agent(name="Support", instructions="Be concise."),
                   model="gpt-4o-mini", playbook="support_memory.json")
out = agent.run_and_learn("Cancel #C99", signal="verify identity first")
agent.save()
```
"""

from typing import TYPE_CHECKING

from .baselines import MonolithicRewriteAgent, StaticAgent
from .config import ACEConfig
from .delta import DeltaContext, DeltaOp, DeltaOperation, MergeResult, apply_delta
from .engine import ACE, FeedbackFn, RunResult, StepRecord
from .feedback import Feedback
from .llm import LLM, OpenAILLM, SimulatedLLM
from .playbook import DEFAULT_SECTIONS, Bullet, Playbook
from .refine import grow_and_refine, make_openai_embedder
from .roles import Curator, Generation, Generator, Reflection, Reflector
from .tasks import Sample, Task, TeachingEnvironment, build_teaching_task

# The OpenAI Agents SDK integration is re-exported lazily (see ``__getattr__``)
# so that ``import ace`` never requires the optional ``openai-agents`` package.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .integrations.openai_agents import ACEAgent, ACERunOutput, wrap_agent

__version__ = "0.2.0"

# Names served on demand by ``__getattr__`` from the agents integration.
_LAZY_AGENTS = {"ACEAgent", "ACERunOutput", "wrap_agent", "playbook_instructions"}

__all__ = [
    "ACE",
    "ACEConfig",
    "RunResult",
    "StepRecord",
    "FeedbackFn",
    "Playbook",
    "Bullet",
    "DEFAULT_SECTIONS",
    "DeltaContext",
    "DeltaOperation",
    "DeltaOp",
    "MergeResult",
    "apply_delta",
    "Generator",
    "Reflector",
    "Curator",
    "Generation",
    "Reflection",
    "Feedback",
    "LLM",
    "OpenAILLM",
    "SimulatedLLM",
    "grow_and_refine",
    "make_openai_embedder",
    "Sample",
    "Task",
    "TeachingEnvironment",
    "build_teaching_task",
    "StaticAgent",
    "MonolithicRewriteAgent",
    # OpenAI Agents SDK integration (lazily imported).
    "wrap_agent",
    "ACEAgent",
    "ACERunOutput",
    "playbook_instructions",
]


def __getattr__(name: str):
    """Lazily expose the OpenAI Agents SDK integration at the top level.

    Importing the *class objects* does not require the SDK; only constructing an
    :class:`~ace.integrations.openai_agents.ACEAgent` does.
    """
    if name in _LAZY_AGENTS:
        from . import integrations  # noqa: F401  (ensure subpackage import)
        from .integrations import openai_agents

        return getattr(openai_agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_AGENTS))
