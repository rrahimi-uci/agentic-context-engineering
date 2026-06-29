"""The three ACE roles: Generator, Reflector, Curator (§3, Figure 4).

ACE deliberately splits responsibilities across three specialized roles instead
of overloading one model:

* **Generator** — produces a reasoning trajectory and an answer for a query,
  and flags which playbook bullets were helpful or misleading.
* **Reflector** — critiques the trajectory against feedback and distills
  concrete, reusable *insights* (optionally over several refinement rounds).
* **Curator** — turns those insights into compact **delta operations** that are
  merged into the playbook by deterministic logic (never a full rewrite).

Each role works with any :class:`~ace.llm.LLM` backend. When the backend is the
offline :class:`~ace.llm.SimulatedLLM`, the roles delegate to its teaching
environment so the demos and tests run deterministically without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .delta import DeltaContext, DeltaOp, DeltaOperation
from .feedback import Feedback
from .llm import LLM, SimulatedLLM
from .playbook import Playbook
from .tasks import Sample


# --------------------------------------------------------------------------- #
# Data passed between roles
# --------------------------------------------------------------------------- #
@dataclass
class Generation:
    answer: str
    reasoning: str = ""
    used_bullet_ids: List[str] = field(default_factory=list)
    helpful_ids: List[str] = field(default_factory=list)
    harmful_ids: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class Reflection:
    insights: List[dict] = field(default_factory=list)  # {content, section, tags}
    helpful_ids: List[str] = field(default_factory=list)
    harmful_ids: List[str] = field(default_factory=list)
    diagnosis: str = ""


# --------------------------------------------------------------------------- #
# Prompts (used by real LLM backends)
# --------------------------------------------------------------------------- #
GENERATOR_SYSTEM = """You are the Generator in the ACE framework.
You solve the user's query using the provided PLAYBOOK of accumulated strategies,
domain concepts, and known pitfalls. Reason step by step, then give a final answer.
Also report which playbook bullets (by their [ctx-...] id) were helpful or misleading.

Return JSON:
{
  "reasoning": "<concise step-by-step reasoning>",
  "answer": "<final answer>",
  "helpful_ids": ["ctx-..."],
  "harmful_ids": ["ctx-..."]
}"""

REFLECTOR_SYSTEM = """You are the Reflector in the ACE framework.
You critique a Generator trajectory against the available feedback and distill
CONCRETE, REUSABLE lessons. Prefer specific, durable insights (domain rules,
tool-use gotchas, failure modes) over vague advice. Do NOT summarize away detail.

Return JSON:
{
  "diagnosis": "<what went right/wrong and why>",
  "insights": [
    {"content": "<a single reusable lesson>", "section": "domain_concepts|strategies|common_mistakes|tool_usage|formatting", "tags": ["..."]}
  ],
  "helpful_ids": ["ctx-..."],
  "harmful_ids": ["ctx-..."]
}"""

CURATOR_SYSTEM = """You are the Curator in the ACE framework.
You convert the Reflector's insights into a small set of localized DELTA
operations over the playbook. NEVER rewrite the whole playbook. Add genuinely
new lessons, update a bullet only to sharpen it, and remove only clearly wrong
or obsolete bullets. Keep each bullet atomic.

Return JSON:
{
  "operations": [
    {"op": "ADD", "section": "domain_concepts", "content": "...", "tags": ["..."]},
    {"op": "UPDATE", "target_id": "ctx-...", "content": "..."},
    {"op": "REMOVE", "target_id": "ctx-..."}
  ]
}"""


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class Generator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def generate(self, sample: Sample, playbook: Playbook) -> Generation:
        if isinstance(self.llm, SimulatedLLM):
            ans, reasoning, used, helpful, harmful = self.llm.env.generate(sample, playbook)
            self.llm.num_calls += 1
            return Generation(
                answer=ans,
                reasoning=reasoning,
                used_bullet_ids=used,
                helpful_ids=helpful,
                harmful_ids=harmful,
            )
        user = (
            f"PLAYBOOK:\n{playbook.render()}\n\n"
            f"QUERY:\n{sample.question}\n\n"
            "Solve it using the playbook."
        )
        data = self.llm.complete_json(GENERATOR_SYSTEM, user)
        return Generation(
            answer=str(data.get("answer", "")).strip(),
            reasoning=str(data.get("reasoning", "")),
            helpful_ids=list(data.get("helpful_ids", [])),
            harmful_ids=list(data.get("harmful_ids", [])),
            used_bullet_ids=list(data.get("helpful_ids", [])) + list(data.get("harmful_ids", [])),
            raw=str(data),
        )


# --------------------------------------------------------------------------- #
# Reflector
# --------------------------------------------------------------------------- #
class Reflector:
    def __init__(self, llm: LLM, max_rounds: int = 1) -> None:
        self.llm = llm
        self.max_rounds = max_rounds

    def reflect(
        self,
        sample: Sample,
        generation: Generation,
        feedback: Feedback,
        playbook: Playbook,
    ) -> Reflection:
        if isinstance(self.llm, SimulatedLLM):
            correct = feedback.correct
            if correct is None and feedback.ground_truth is not None:
                correct = generation.answer.strip().lower() == feedback.ground_truth.strip().lower()
            insights, diagnosis = self.llm.env.reflect(sample, bool(correct), feedback.has_label)
            self.llm.num_calls += 1
            return Reflection(
                insights=insights,
                helpful_ids=generation.helpful_ids,
                harmful_ids=generation.harmful_ids,
                diagnosis=diagnosis,
            )

        fb_lines = []
        if feedback.correct is not None:
            fb_lines.append(f"Correct: {feedback.correct}")
        if feedback.ground_truth:
            fb_lines.append(f"Ground truth: {feedback.ground_truth}")
        if feedback.signal:
            fb_lines.append(f"Execution feedback: {feedback.signal}")
        feedback_str = "\n".join(fb_lines) or "(no explicit feedback; use your judgment)"

        user = (
            f"PLAYBOOK:\n{playbook.render()}\n\n"
            f"QUERY:\n{sample.question}\n\n"
            f"GENERATOR REASONING:\n{generation.reasoning}\n\n"
            f"GENERATOR ANSWER:\n{generation.answer}\n\n"
            f"FEEDBACK:\n{feedback_str}\n\n"
            "Distill concrete, reusable lessons."
        )
        reflection: Optional[Reflection] = None
        for _ in range(max(1, self.max_rounds)):
            data = self.llm.complete_json(REFLECTOR_SYSTEM, user)
            reflection = Reflection(
                insights=list(data.get("insights", [])),
                helpful_ids=list(data.get("helpful_ids", [])) or generation.helpful_ids,
                harmful_ids=list(data.get("harmful_ids", [])) or generation.harmful_ids,
                diagnosis=str(data.get("diagnosis", "")),
            )
            # Iterative refinement: feed the prior insights back in.
            if reflection.insights:
                user += f"\n\nPRIOR INSIGHTS (refine, don't repeat):\n{reflection.insights}"
            else:
                break
        return reflection or Reflection()


# --------------------------------------------------------------------------- #
# Curator
# --------------------------------------------------------------------------- #
class Curator:
    """Turns reflection insights into localized delta operations.

    With a real LLM backend the Curator *calls the model* (``CURATOR_SYSTEM``)
    so it can propose model-driven ``ADD`` / ``UPDATE`` / ``REMOVE`` edits
    against the existing playbook — e.g. sharpening a bullet or removing one
    that proved wrong. If the model returns nothing usable (or ``use_llm`` is
    off, or the backend is the offline :class:`SimulatedLLM`), it falls back to
    a deterministic build that turns each insight into an ``ADD`` — so a lesson
    is never silently dropped.
    """

    def __init__(self, llm: LLM, use_llm: bool = True) -> None:
        self.llm = llm
        self.use_llm = use_llm

    def curate(
        self,
        sample: Sample,
        generation: Generation,
        reflection: Reflection,
        playbook: Playbook,
    ) -> DeltaContext:
        ops: List[DeltaOperation] = []

        use_model = self.use_llm and not isinstance(self.llm, SimulatedLLM)
        if use_model and reflection.insights:
            user = (
                f"CURRENT PLAYBOOK (edit by id where appropriate):\n{playbook.render()}\n\n"
                f"DIAGNOSIS:\n{reflection.diagnosis}\n\n"
                f"NEW INSIGHTS TO INTEGRATE:\n{reflection.insights}\n\n"
                "Emit a minimal set of delta operations."
            )
            try:
                data = self.llm.complete_json(CURATOR_SYSTEM, user)
                for od in data.get("operations", []):
                    op = DeltaOperation.from_dict(od)
                    # Drop ADD/UPDATE with empty content; keep REMOVE.
                    if op.op is not DeltaOp.REMOVE and not op.content.strip():
                        continue
                    ops.append(op)
            except Exception:
                ops = []

        # Deterministic fallback: never drop a distilled lesson.
        if not ops:
            for ins in reflection.insights:
                content = str(ins.get("content", "")).strip()
                if not content:
                    continue
                ops.append(
                    DeltaOperation(
                        op=DeltaOp.ADD,
                        section=str(ins.get("section", "strategies")),
                        content=content,
                        tags=list(ins.get("tags", [])),
                    )
                )

        if isinstance(self.llm, SimulatedLLM):
            self.llm.num_calls += 1
        return DeltaContext(
            operations=ops,
            helpful_ids=reflection.helpful_ids,
            harmful_ids=reflection.harmful_ids,
        )
