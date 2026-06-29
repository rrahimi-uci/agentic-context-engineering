"""Baselines for comparison — including the context-collapse failure mode.

The headline qualitative claim of the paper (Figure 2) is **context collapse**:
when an LLM is asked to *fully rewrite* the accumulated context at each step, a
large context can suddenly compress into a short, lossy summary, erasing
hard-won knowledge and crashing accuracy.

:class:`MonolithicRewriteAgent` reproduces that dynamic in the offline teaching
environment so the demos can show — with numbers — why ACE's incremental delta
updates matter. :class:`StaticAgent` is the no-adaptation base model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List

from .engine import RunResult, StepRecord
from .llm import SimulatedLLM
from .playbook import Bullet, Playbook
from .tasks import Task


class StaticAgent:
    """No adaptation — the base model evaluated directly (paper's 'Base LLM')."""

    def __init__(self, llm: SimulatedLLM) -> None:
        self.llm = llm
        self.playbook = Playbook()

    def run(self, task: Task) -> RunResult:
        history: List[StepRecord] = []
        for i, s in enumerate(task.samples, 1):
            ans, *_ = self.llm.env.generate(s, self.playbook)
            history.append(
                StepRecord(
                    step=i,
                    phase="base",
                    sample_id=s.id,
                    question=s.question,
                    prediction=ans,
                    ground_truth=s.answer or None,
                    correct=task.evaluate(ans, s) if s.answer else None,
                    delta={},
                    merge={},
                    refine={},
                    playbook_size=0,
                    playbook_tokens=0,
                )
            )
        return RunResult(history=history, playbook=self.playbook)


@dataclass
class MonolithicRewriteAgent:
    """Accumulate context by full LLM rewrites — and suffer context collapse.

    Like ACE it learns rules from feedback, but instead of localized delta
    merges it conceptually re-emits the *entire* context each step. With
    probability ``collapse_prob`` a rewrite "collapses": it compresses the
    context down to only the most recent ``keep_on_collapse`` items, dropping
    older knowledge (and the accuracy that depended on it).
    """

    llm: SimulatedLLM
    collapse_prob: float = 0.18
    keep_on_collapse: int = 2
    playbook: Playbook = field(default_factory=Playbook)

    def _collapses(self, step: int) -> bool:
        # Deterministic pseudo-random collapse based on the step (reproducible).
        h = int(hashlib.sha256(f"collapse:{step}".encode()).hexdigest(), 16)
        return (h % 1000) / 1000.0 < self.collapse_prob

    def run(self, task: Task, use_labels: bool = True) -> RunResult:
        history: List[StepRecord] = []
        for i, s in enumerate(task.samples, 1):
            # Predict with current (monolithic) context.
            ans, _reason, _used, _hp, _hm = self.llm.env.generate(s, self.playbook)
            correct = task.evaluate(ans, s) if s.answer else None

            # Reflect to extract the lesson (same env as ACE — fair comparison).
            insights, _diag = self.llm.env.reflect(s, bool(correct), use_labels and bool(s.answer))
            for ins in insights:
                self.playbook.add(
                    Bullet(
                        content=str(ins["content"]),
                        section=str(ins["section"]),
                        tags=list(ins.get("tags", [])),
                        created_at_step=i,
                    )
                )

            # Monolithic rewrite step — risks collapse.
            collapsed = False
            if self._collapses(i) and len(self.playbook) > self.keep_on_collapse:
                kept = self.playbook.bullets[-self.keep_on_collapse :]
                self.playbook = Playbook()
                for b in kept:
                    self.playbook.add(b)
                collapsed = True

            history.append(
                StepRecord(
                    step=i,
                    phase="monolithic",
                    sample_id=s.id,
                    question=s.question,
                    prediction=ans,
                    ground_truth=s.answer or None,
                    correct=correct,
                    delta={},
                    merge={},
                    refine={"collapsed": collapsed},
                    playbook_size=len(self.playbook),
                    playbook_tokens=self.playbook.approx_tokens(),
                    diagnosis="CONTEXT COLLAPSE" if collapsed else "",
                )
            )
        return RunResult(history=history, playbook=self.playbook)
