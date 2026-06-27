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
"""

from .baselines import MonolithicRewriteAgent, StaticAgent
from .config import ACEConfig
from .delta import DeltaContext, DeltaOp, DeltaOperation, MergeResult, apply_delta
from .engine import ACE, FeedbackFn, RunResult, StepRecord
from .feedback import Feedback
from .llm import LLM, OpenAILLM, SimulatedLLM
from .playbook import Bullet, Playbook, DEFAULT_SECTIONS
from .refine import grow_and_refine, make_openai_embedder
from .roles import Curator, Generation, Generator, Reflection, Reflector
from .tasks import Sample, Task, TeachingEnvironment, build_teaching_task

__version__ = "0.1.0"

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
]
