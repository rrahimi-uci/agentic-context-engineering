"""The ACE engine: the generate → reflect → curate → merge → refine loop.

This ties the three roles, deterministic delta merging, and grow-and-refine into
the two adaptation regimes from the paper:

* :meth:`ACE.adapt_offline` — multi-epoch optimization over a training split
  (e.g. system-prompt optimization), optionally with ground-truth labels.
* :meth:`ACE.adapt_online` — sequential test-time adaptation: for each sample
  the agent first *predicts* with the current context, then *updates* it.

Every step emits a :class:`StepRecord`, and a callback hook lets a live
visualizer watch the run unfold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import ACEConfig
from .delta import DeltaContext, MergeResult, apply_delta
from .feedback import Feedback
from .llm import LLM
from .playbook import Playbook
from .refine import Embedder, RefineResult, grow_and_refine
from .roles import Curator, Generation, Generator, Reflection, Reflector
from .tasks import Sample, Task


@dataclass
class StepRecord:
    """Everything that happened on one adaptation step (for analysis/viz)."""

    step: int
    phase: str                 # "offline-epochN" or "online"
    sample_id: str
    question: str
    prediction: str
    ground_truth: Optional[str]
    correct: Optional[bool]
    delta: dict                # serialized DeltaContext
    merge: dict                # added/updated/removed ids
    refine: dict               # dedup/pruned ids
    playbook_size: int
    playbook_tokens: int
    diagnosis: str = ""
    latency_s: float = 0.0


@dataclass
class RunResult:
    """Aggregate result of an adaptation run."""

    history: List[StepRecord] = field(default_factory=list)
    playbook: Optional[Playbook] = None

    # Convenience metrics ------------------------------------------------ #
    @property
    def accuracy(self) -> float:
        graded = [r for r in self.history if r.correct is not None]
        if not graded:
            return 0.0
        return 100.0 * sum(1 for r in graded if r.correct) / len(graded)

    def accuracy_in_window(self, start: int, end: Optional[int] = None) -> float:
        rows = self.history[start:end]
        graded = [r for r in rows if r.correct is not None]
        if not graded:
            return 0.0
        return 100.0 * sum(1 for r in graded if r.correct) / len(graded)

    @property
    def growth_curve(self) -> List[int]:
        return [r.playbook_size for r in self.history]

    @property
    def token_curve(self) -> List[int]:
        return [r.playbook_tokens for r in self.history]

    def summary(self) -> dict:
        graded = [r for r in self.history if r.correct is not None]
        return {
            "steps": len(self.history),
            "graded": len(graded),
            "accuracy": round(self.accuracy, 2),
            "final_playbook_size": self.playbook.stats()["num_bullets"] if self.playbook else 0,
            "final_playbook_tokens": self.playbook.approx_tokens() if self.playbook else 0,
        }


StepCallback = Callable[[StepRecord], None]

# A user-supplied hook that turns a (sample, generation) pair into Feedback.
# This is the extension point for *custom* and *label-free* tasks: plug in your
# own execution signals (test pass/fail, API errors, a reward function, an
# LLM-as-judge, ...) instead of relying on ground-truth labels.
FeedbackFn = Callable[[Sample, Generation], Feedback]


class ACE:
    """Orchestrates Agentic Context Engineering over a playbook.

    Parameters
    ----------
    llm:
        Backend shared by all three roles (matching the paper's fair-comparison
        setup). You may pass distinct backends via ``generator_llm`` etc.
    config:
        :class:`~ace.config.ACEConfig`. Sensible paper-aligned defaults.
    playbook:
        An existing playbook to continue adapting (otherwise a fresh one).
    embedder:
        Optional embedding function for semantic de-duplication.
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[ACEConfig] = None,
        playbook: Optional[Playbook] = None,
        embedder: Optional[Embedder] = None,
        generator_llm: Optional[LLM] = None,
        reflector_llm: Optional[LLM] = None,
        curator_llm: Optional[LLM] = None,
    ) -> None:
        self.config = config or ACEConfig()
        self.playbook = playbook or Playbook(self.config.sections)
        self.embedder = embedder
        self.generator = Generator(generator_llm or llm)
        self.reflector = Reflector(reflector_llm or llm, max_rounds=self.config.reflector_max_rounds)
        self.curator = Curator(curator_llm or llm, use_llm=self.config.curator_use_llm)
        self._step = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, sample: Sample) -> Generation:
        """Run only the Generator (inference with the current playbook)."""
        return self.generator.generate(sample, self.playbook)

    def step(
        self,
        sample: Sample,
        feedback: Feedback,
        phase: str = "online",
        callback: Optional[StepCallback] = None,
        generation: Optional[Generation] = None,
    ) -> StepRecord:
        """One full adaptation step on a single sample.

        If ``generation`` is provided it is reused (online setting: predict
        first, then learn from the *same* trajectory).
        """
        t0 = time.time()
        self._step += 1

        gen = generation or self.generator.generate(sample, self.playbook)

        # Resolve correctness from feedback / ground truth.
        correct = feedback.correct
        if correct is None and feedback.ground_truth is not None:
            correct = gen.answer.strip().lower() == feedback.ground_truth.strip().lower()

        reflection = self.reflector.reflect(sample, gen, feedback, self.playbook)
        delta = self.curator.curate(sample, gen, reflection, self.playbook)
        merge = apply_delta(self.playbook, delta, step=self._step)
        refine = self._maybe_refine()

        rec = StepRecord(
            step=self._step,
            phase=phase,
            sample_id=sample.id,
            question=sample.question,
            prediction=gen.answer,
            ground_truth=feedback.ground_truth,
            correct=correct,
            delta=delta.to_dict(),
            merge={"added": merge.added, "updated": merge.updated, "removed": merge.removed,
                   "helpful_marked": merge.helpful_marked, "harmful_marked": merge.harmful_marked},
            refine={"deduped": refine.deduped, "pruned": refine.pruned},
            playbook_size=len(self.playbook),
            playbook_tokens=self.playbook.approx_tokens(),
            diagnosis=reflection.diagnosis,
            latency_s=round(time.time() - t0, 4),
        )
        if callback:
            callback(rec)
        return rec

    def adapt_offline(
        self,
        task: Task,
        callback: Optional[StepCallback] = None,
        feedback_fn: Optional[FeedbackFn] = None,
    ) -> RunResult:
        """Multi-epoch offline adaptation over a (training) task.

        Mirrors offline context optimization: revisit the same samples across
        epochs to progressively strengthen the playbook.

        Pass ``feedback_fn`` to supply custom / label-free feedback for each
        ``(sample, generation)`` instead of relying on ``sample.answer``.
        """
        history: List[StepRecord] = []
        for epoch in range(self.config.epochs):
            for sample in task.samples:
                gen = self.generator.generate(sample, self.playbook)
                fb = self._build_feedback(sample, task, generation=gen, feedback_fn=feedback_fn)
                rec = self.step(sample, fb, phase=f"offline-e{epoch + 1}",
                                callback=callback, generation=gen)
                history.append(rec)
        return RunResult(history=history, playbook=self.playbook)

    def adapt_online(
        self,
        task: Task,
        callback: Optional[StepCallback] = None,
        feedback_fn: Optional[FeedbackFn] = None,
    ) -> RunResult:
        """Sequential online adaptation: predict, then learn, per sample.

        Pass ``feedback_fn`` to supply custom / label-free feedback for each
        ``(sample, generation)`` (e.g. environment signals, a reward function).
        """
        history: List[StepRecord] = []
        for sample in task.samples:
            gen = self.generator.generate(sample, self.playbook)  # predict first
            fb = self._build_feedback(sample, task, generation=gen, feedback_fn=feedback_fn)
            rec = self.step(sample, fb, phase="online", callback=callback, generation=gen)
            history.append(rec)
        return RunResult(history=history, playbook=self.playbook)

    def evaluate(self, task: Task) -> RunResult:
        """Inference-only pass (no adaptation) — useful as a baseline."""
        history: List[StepRecord] = []
        for sample in task.samples:
            self._step += 1
            gen = self.generator.generate(sample, self.playbook)
            correct = task.evaluate(gen.answer, sample) if sample.answer else None
            history.append(
                StepRecord(
                    step=self._step, phase="eval", sample_id=sample.id,
                    question=sample.question, prediction=gen.answer,
                    ground_truth=sample.answer or None, correct=correct,
                    delta={}, merge={}, refine={},
                    playbook_size=len(self.playbook),
                    playbook_tokens=self.playbook.approx_tokens(),
                )
            )
        return RunResult(history=history, playbook=self.playbook)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_feedback(
        self,
        sample: Sample,
        task: Task,
        generation: Optional[Generation] = None,
        feedback_fn: Optional[FeedbackFn] = None,
    ) -> Feedback:
        # 1) Caller-supplied feedback wins (custom / label-free path).
        if feedback_fn is not None:
            return feedback_fn(sample, generation)
        prediction = generation.answer if generation is not None else None
        # 2) Label-free with no hook: nothing reliable to learn from.
        if not self.config.use_labels or not sample.answer:
            return Feedback(correct=None, ground_truth=None, signal="")
        # 3) Labeled: grade against the task scorer.
        correct = task.evaluate(prediction, sample) if prediction is not None else None
        return Feedback(correct=correct, ground_truth=sample.answer, signal="")

    def _maybe_refine(self) -> RefineResult:
        cfg = self.config
        do_refine = False
        if cfg.lazy_refine_token_budget is not None:
            do_refine = self.playbook.approx_tokens() > cfg.lazy_refine_token_budget
        elif cfg.refine_every and self._step % cfg.refine_every == 0:
            do_refine = True
        if not do_refine:
            return RefineResult(deduped=[], pruned=[])
        return grow_and_refine(
            self.playbook,
            dedup_threshold=cfg.dedup_threshold,
            prune_harmful=cfg.prune_harmful,
            harmful_margin=cfg.harmful_margin,
            embedder=self.embedder,
        )
