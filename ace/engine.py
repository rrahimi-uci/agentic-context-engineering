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
from .delta import apply_delta
from .feedback import Feedback
from .llm import LLM, OpenAILLM
from .playbook import Playbook
from .refine import Embedder, RefineResult, grow_and_refine
from .roles import Curator, Generation, Generator, Reflector
from .tasks import Sample, Task


@dataclass
class StepRecord:
    """Everything that happened on one adaptation step (for analysis/viz)."""

    step: int
    phase: str  # "offline-epochN" or "online"
    sample_id: str
    question: str
    prediction: str
    ground_truth: Optional[str]
    correct: Optional[bool]
    delta: dict  # serialized DeltaContext
    merge: dict  # added/updated/removed ids
    refine: dict  # dedup/pruned ids
    playbook_size: int
    playbook_tokens: int
    diagnosis: str = ""
    latency_s: float = 0.0
    # LLM usage attributable to this step (0 under the offline SimulatedLLM).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # served from the provider's prompt cache
    llm_calls: int = 0


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

    # LLM usage / cost (summed over the run; nonzero with a real backend) ---- #
    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.history)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.history)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cached_prompt_tokens(self) -> int:
        """Prompt tokens served from the provider's cache (e.g. OpenAI's
        automatic prefix caching of the static system + playbook prefix)."""
        return sum(r.cached_prompt_tokens for r in self.history)

    @property
    def total_llm_calls(self) -> int:
        return sum(r.llm_calls for r in self.history)

    def summary(self) -> dict:
        graded = [r for r in self.history if r.correct is not None]
        return {
            "steps": len(self.history),
            "graded": len(graded),
            "accuracy": round(self.accuracy, 2),
            "final_playbook_size": self.playbook.stats()["num_bullets"] if self.playbook else 0,
            "final_playbook_tokens": self.playbook.approx_tokens() if self.playbook else 0,
            "llm_calls": self.total_llm_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.total_cached_prompt_tokens,
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
        self.generator = Generator(generator_llm or llm)
        self.reflector = Reflector(
            reflector_llm or llm, max_rounds=self.config.reflector_max_rounds
        )
        self.curator = Curator(curator_llm or llm, use_llm=self.config.curator_use_llm)
        # Use the caller's embedder, else auto-wire a semantic one from an OpenAI
        # backend (for grow-and-refine dedup); otherwise stay lexical.
        self.embedder = embedder if embedder is not None else self._auto_embedder()
        self._step = 0

    def _auto_embedder(self) -> Optional[Embedder]:
        if not self.config.auto_embedder:
            return None
        for role_llm in (self.generator.llm, self.reflector.llm, self.curator.llm):
            if isinstance(role_llm, OpenAILLM):
                try:
                    return role_llm.embedder()
                except Exception:  # pragma: no cover - defensive; stay lexical
                    return None
        return None

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
        usage_before = self._usage_snapshot()

        gen = generation or self.generator.generate(sample, self.playbook)

        # Resolve correctness from feedback / ground truth.
        correct = feedback.correct
        if correct is None and feedback.ground_truth is not None:
            correct = gen.answer.strip().lower() == feedback.ground_truth.strip().lower()

        reflection = self.reflector.reflect(sample, gen, feedback, self.playbook)
        delta = self.curator.curate(sample, gen, reflection, self.playbook)
        merge = apply_delta(self.playbook, delta, step=self._step)
        refine = self._maybe_refine()

        usage = self._usage_delta(usage_before, self._usage_snapshot())
        rec = StepRecord(
            step=self._step,
            phase=phase,
            sample_id=sample.id,
            question=sample.question,
            prediction=gen.answer,
            ground_truth=feedback.ground_truth,
            correct=correct,
            delta=delta.to_dict(),
            merge={
                "added": merge.added,
                "updated": merge.updated,
                "removed": merge.removed,
                "helpful_marked": merge.helpful_marked,
                "harmful_marked": merge.harmful_marked,
            },
            refine={"deduped": refine.deduped, "pruned": refine.pruned},
            playbook_size=len(self.playbook),
            playbook_tokens=self.playbook.approx_tokens(),
            diagnosis=reflection.diagnosis,
            latency_s=round(time.time() - t0, 4),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_prompt_tokens=usage["cached_prompt_tokens"],
            llm_calls=usage["llm_calls"],
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
                u0 = self._usage_snapshot()
                gen = self.generator.generate(sample, self.playbook)
                gen_usage = self._usage_delta(u0, self._usage_snapshot())
                fb = self._build_feedback(sample, task, generation=gen, feedback_fn=feedback_fn)
                rec = self.step(
                    sample, fb, phase=f"offline-e{epoch + 1}", callback=callback, generation=gen
                )
                self._apply_usage(rec, gen_usage)  # fold in the pre-step generation cost
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
            u0 = self._usage_snapshot()
            gen = self.generator.generate(sample, self.playbook)  # predict first
            gen_usage = self._usage_delta(u0, self._usage_snapshot())
            fb = self._build_feedback(sample, task, generation=gen, feedback_fn=feedback_fn)
            rec = self.step(sample, fb, phase="online", callback=callback, generation=gen)
            self._apply_usage(rec, gen_usage)  # fold in the pre-step generation cost
            history.append(rec)
        return RunResult(history=history, playbook=self.playbook)

    def evaluate(self, task: Task, max_workers: int = 1) -> RunResult:
        """Inference-only pass (no adaptation) — useful as a baseline.

        The playbook never changes here, so each sample's generation is
        independent: ``max_workers > 1`` runs them concurrently for a faster pass
        with **identical results**. (In parallel mode LLM-usage counts are split
        evenly across records — run-level totals stay correct; per-record values
        are approximate.)
        """
        samples = task.samples
        if max_workers > 1 and len(samples) > 1:
            from concurrent.futures import ThreadPoolExecutor

            u0 = self._usage_snapshot()
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                gens = list(pool.map(lambda s: self.generator.generate(s, self.playbook), samples))
            shares = self._split_usage(self._usage_delta(u0, self._usage_snapshot()), len(samples))
            history = [
                self._eval_record(sample, gens[i], shares[i], task)
                for i, sample in enumerate(samples)
            ]
        else:
            history = []
            for sample in samples:
                u0 = self._usage_snapshot()
                gen = self.generator.generate(sample, self.playbook)
                usage = self._usage_delta(u0, self._usage_snapshot())
                history.append(self._eval_record(sample, gen, usage, task))
        return RunResult(history=history, playbook=self.playbook)

    def _eval_record(self, sample: Sample, gen: Generation, usage: dict, task: Task) -> StepRecord:
        self._step += 1
        correct = task.evaluate(gen.answer, sample) if sample.answer else None
        return StepRecord(
            step=self._step,
            phase="eval",
            sample_id=sample.id,
            question=sample.question,
            prediction=gen.answer,
            ground_truth=sample.answer or None,
            correct=correct,
            delta={},
            merge={},
            refine={},
            playbook_size=len(self.playbook),
            playbook_tokens=self.playbook.approx_tokens(),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_prompt_tokens=usage["cached_prompt_tokens"],
            llm_calls=usage["llm_calls"],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _usage_snapshot(self) -> dict:
        """Cumulative LLM usage across the (possibly distinct) role backends.

        Reads optional counters off each backend (``prompt_tokens``,
        ``completion_tokens``, ``cached_prompt_tokens``, ``num_calls``); missing
        counters are treated as 0 so any custom :class:`~ace.llm.LLM` works.
        Shared backends are counted once.
        """
        seen: set = set()
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "llm_calls": 0,
        }
        for llm in (self.generator.llm, self.reflector.llm, self.curator.llm):
            if id(llm) in seen:
                continue
            seen.add(id(llm))
            total["prompt_tokens"] += getattr(llm, "prompt_tokens", 0)
            total["completion_tokens"] += getattr(llm, "completion_tokens", 0)
            total["cached_prompt_tokens"] += getattr(llm, "cached_prompt_tokens", 0)
            total["llm_calls"] += getattr(llm, "num_calls", 0)
        return total

    @staticmethod
    def _usage_delta(before: dict, after: dict) -> dict:
        return {k: after[k] - before[k] for k in after}

    @staticmethod
    def _split_usage(total: dict, n: int) -> List[dict]:
        """Split a usage total into ``n`` near-equal parts (sum is preserved)."""
        if n <= 0:
            return []
        parts = [{k: total[k] // n for k in total} for _ in range(n)]
        for k in total:
            for i in range(total[k] - (total[k] // n) * n):  # spread the remainder
                parts[i][k] += 1
        return parts

    @staticmethod
    def _apply_usage(rec: StepRecord, usage: dict) -> None:
        rec.prompt_tokens += usage["prompt_tokens"]
        rec.completion_tokens += usage["completion_tokens"]
        rec.cached_prompt_tokens += usage["cached_prompt_tokens"]
        rec.llm_calls += usage["llm_calls"]

    def _build_feedback(
        self,
        sample: Sample,
        task: Task,
        generation: Optional[Generation] = None,
        feedback_fn: Optional[FeedbackFn] = None,
    ) -> Feedback:
        # 1) Caller-supplied feedback wins (custom / label-free path).
        if feedback_fn is not None:
            gen = generation or self.generator.generate(sample, self.playbook)
            return feedback_fn(sample, gen)
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
