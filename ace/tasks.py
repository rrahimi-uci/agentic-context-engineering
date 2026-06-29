"""Task samples and the offline *teaching environment*.

A :class:`Sample` is one query the agent must answer. A :class:`Task` is a
collection of samples plus an ``evaluate`` function.

:class:`TeachingEnvironment` is a self-contained, deterministic benchmark used
by the demos and tests. Each question is tied to a hidden *domain rule*. A base
model only "innately knows" a fraction of those rules; for the rest it must
learn the rule from the playbook. This makes ACE's improvement **real and
measurable** while requiring no API key — perfect for reproducible demos and CI.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, cast


@dataclass
class Sample:
    """A single query/answer item."""

    id: str
    question: str
    answer: str = ""
    concept: str = ""  # hidden rule key (teaching env); free in real tasks
    rule_text: str = ""  # the canonical lesson for this concept
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Task:
    """A named collection of samples plus a scorer."""

    name: str
    samples: List[Sample]
    evaluate: Callable[[str, Sample], bool]
    description: str = ""

    def split(self, train_frac: float = 0.5, seed: int = 0):
        rng = random.Random(seed)
        idx = list(range(len(self.samples)))
        rng.shuffle(idx)
        cut = int(len(idx) * train_frac)
        train = [self.samples[i] for i in idx[:cut]]
        test = [self.samples[i] for i in idx[cut:]]
        return (
            Task(self.name + "-train", train, self.evaluate, self.description),
            Task(self.name + "-test", test, self.evaluate, self.description),
        )


# --------------------------------------------------------------------------- #
# Teaching environment: a deterministic, offline benchmark.
# --------------------------------------------------------------------------- #

# A compact catalogue of domain rules across a few areas. Each is phrased as a
# crisp, reusable lesson — exactly the kind of bullet ACE should accumulate.
_RULE_BANK: List[Dict[str, Any]] = [
    {
        "concept": "reserves_multi_property",
        "rule": "For a 2-unit investment property with 4 financed properties, minimum required reserves are 12 months (not 2 or 6).",
        "q": "Minimum months of reserves for a 2-unit investment property with four financed properties, all retained post-closing?",
        "options": ["2 months", "6 months", "12 months", "8 months"],
        "answer": "12 months",
    },
    {
        "concept": "ipc_high_ltv",
        "rule": "Interested-party contributions (IPC) for a primary residence with LTV > 90% are capped at 3% of the purchase price.",
        "q": "Max interested party contribution for a primary residence at 95% LTV?",
        "options": ["2%", "3%", "6%", "9%"],
        "answer": "3%",
    },
    {
        "concept": "virtual_currency_funds",
        "rule": "Virtual currency must be converted to USD and deposited in a U.S. depository institution before closing, with documentation of conversion and source.",
        "q": "Requirement for virtual currency used toward down payment to be eligible for closing?",
        "options": [
            "Convert to USD before loan application",
            "Held in a regulated exchange at closing",
            "Convert to USD and deposit in a U.S. institution before closing, with documentation",
            "Use directly with an exchange letter",
        ],
        "answer": "Convert to USD and deposit in a U.S. institution before closing, with documentation",
    },
    {
        "concept": "gift_funds_primary",
        "rule": "Gift funds are allowed for the entire down payment on a primary residence; no minimum borrower contribution is required for a 1-unit primary residence.",
        "q": "Minimum borrower own-funds contribution when using gift funds for a 1-unit primary residence?",
        "options": ["5%", "3%", "0% (none required)", "10%"],
        "answer": "0% (none required)",
    },
    {
        "concept": "dti_max_manual",
        "rule": "For manually underwritten loans, the maximum total debt-to-income (DTI) ratio is generally 36%, extendable to 45% with strong compensating factors.",
        "q": "Maximum total DTI for a manually underwritten loan with strong compensating factors?",
        "options": ["36%", "43%", "45%", "50%"],
        "answer": "45%",
    },
    {
        "concept": "appraisal_waiver",
        "rule": "An appraisal waiver is not permitted on cash-out refinances; a full appraisal is required.",
        "q": "Is an appraisal waiver permitted on a cash-out refinance?",
        "options": [
            "Yes, always",
            "Yes, under 80% LTV",
            "No, a full appraisal is required",
            "Only for primary residences",
        ],
        "answer": "No, a full appraisal is required",
    },
    {
        "concept": "self_employed_history",
        "rule": "Self-employed borrowers generally need a two-year history of self-employment; one year may be acceptable with documented prior related experience.",
        "q": "Standard self-employment history required to qualify income?",
        "options": ["6 months", "1 year", "2 years", "5 years"],
        "answer": "2 years",
    },
    {
        "concept": "escrow_high_ltv",
        "rule": "Escrow accounts for taxes and insurance are required when the LTV exceeds 80% on most conventional loans.",
        "q": "When are escrow accounts generally required on a conventional loan?",
        "options": ["LTV > 80%", "LTV > 95%", "Never", "Only for investment properties"],
        "answer": "LTV > 80%",
    },
    {
        "concept": "condo_review",
        "rule": "Established condo projects with limited review require the HOA to carry adequate master insurance and have < 15% of units in arrears on dues.",
        "q": "A limited condo review requires HOA dues arrears below what threshold?",
        "options": ["5%", "10%", "15%", "25%"],
        "answer": "15%",
    },
    {
        "concept": "rate_lock",
        "rule": "A rate lock guarantees the interest rate for a defined period; if it expires before closing, the loan must be re-locked at current market rates.",
        "q": "What happens if a rate lock expires before closing?",
        "options": [
            "Rate stays the same",
            "Loan is denied",
            "Must re-lock at current market rates",
            "Borrower pays a flat fee",
        ],
        "answer": "Must re-lock at current market rates",
    },
    {
        "concept": "income_bonus",
        "rule": "Bonus and overtime income require a two-year average and documented likelihood of continuance to be used for qualifying.",
        "q": "How is bonus income treated for qualifying?",
        "options": [
            "Use most recent year",
            "Two-year average with continuance",
            "Cannot be used",
            "Use highest year",
        ],
        "answer": "Two-year average with continuance",
    },
    {
        "concept": "subordinate_financing",
        "rule": "Combined LTV (CLTV) including subordinate financing generally cannot exceed 95% for a 1-unit primary residence purchase.",
        "q": "Maximum CLTV with subordinate financing for a 1-unit primary residence purchase?",
        "options": ["80%", "90%", "95%", "100%"],
        "answer": "95%",
    },
]


def _letter(options: List[str], answer: str) -> str:
    idx = options.index(answer)
    return f"{chr(65 + idx)}) {answer}"


def _format_question(rule: Dict[str, str]) -> str:
    lines = [rule["q"], ""]
    for i, opt in enumerate(rule["options"]):
        lines.append(f"{chr(65 + i)}) {opt}")
    return "\n".join(lines)


def build_teaching_task(name: str = "mortgage-rules", repeats: int = 2, seed: int = 0) -> Task:
    """Build a deterministic multiple-choice teaching task.

    ``repeats`` controls how many times the rule bank is cycled (with shuffling),
    which lets online adaptation revisit concepts and demonstrate accumulation.
    """
    rng = random.Random(seed)
    samples: List[Sample] = []
    order = list(range(len(_RULE_BANK)))
    for r in range(repeats):
        rng.shuffle(order)
        for j in order:
            rule = _RULE_BANK[j]
            answer = _letter(rule["options"], rule["answer"])
            samples.append(
                Sample(
                    id=f"{rule['concept']}-{r}-{j}",
                    question=_format_question(rule),
                    answer=answer,
                    concept=rule["concept"],
                    rule_text=rule["rule"],
                    metadata={"options": rule["options"]},
                )
            )

    def evaluate(prediction: str, sample: Sample) -> bool:
        pred = (prediction or "").strip().lower()
        gold = sample.answer.strip().lower()
        if pred == gold:
            return True
        # Accept the letter, the text, or "A) text" forms.
        gold_letter = gold.split(")")[0].strip()
        gold_text = gold.split(")", 1)[-1].strip()
        return gold_letter == pred or gold_text in pred or pred in gold_text and len(pred) > 4

    return Task(
        name=name,
        samples=samples,
        evaluate=evaluate,
        description="Mortgage underwriting multiple-choice rules (teaching environment).",
    )


class TeachingEnvironment:
    """Deterministic simulator of a base model + execution feedback.

    Parameters
    ----------
    known_fraction:
        Fraction of concepts the base model gets right *without* any context.
    seed:
        Controls which concepts are innately known and the small amount of noise.
    """

    def __init__(self, known_fraction: float = 0.35, seed: int = 7) -> None:
        self.known_fraction = known_fraction
        self.seed = seed

    # --- which concepts does the base model already know? --------------- #
    def knows(self, concept: str) -> bool:
        h = int(hashlib.sha256(f"{self.seed}:{concept}".encode()).hexdigest(), 16)
        return (h % 1000) / 1000.0 < self.known_fraction

    def _concept_in_playbook(self, concept: str, playbook) -> Optional[str]:
        """Return the id of a bullet that teaches ``concept``, if present."""
        for b in playbook:
            if concept and concept in (b.tags or []):
                return b.id
        return None

    # --- simulated Generator ------------------------------------------- #
    def generate(self, sample: Sample, playbook):
        """Answer ``sample`` given the current playbook.

        Correct iff the base model knows the concept OR a bullet teaches it.
        Returns (answer, reasoning, used_ids, helpful_ids, harmful_ids).
        """
        taught_by = self._concept_in_playbook(sample.concept, playbook)
        knows = self.knows(sample.concept)
        correct = knows or taught_by is not None

        options = cast("List[str]", sample.metadata.get("options", []))
        if correct:
            answer = sample.answer
            reasoning = f"Recognized the relevant rule for '{sample.concept}' " + (
                "from prior knowledge." if knows else "from the playbook."
            )
        else:
            # Deterministically pick a wrong option.
            wrong = [o for o in options if _letter(options, o) != sample.answer]
            h = int(hashlib.sha256(sample.id.encode()).hexdigest(), 16)
            choice = wrong[h % len(wrong)] if wrong else sample.answer
            answer = _letter(options, choice) if options else "unknown"
            reasoning = f"Guessed; no rule available for '{sample.concept}'."

        used = [taught_by] if taught_by else []
        helpful = [taught_by] if (taught_by and correct) else []
        harmful: List[str] = []
        return answer, reasoning, used, helpful, harmful

    # --- simulated Reflector ------------------------------------------- #
    def reflect(self, sample: Sample, correct: bool, has_label: bool):
        """Extract a lesson from the outcome.

        With a label (or a reliable execution signal), a wrong answer yields the
        canonical rule. Without any reliable signal, reflection is unreliable —
        mirroring the paper's finding that ACE degrades without good feedback.
        """
        insights: List[Dict[str, object]] = []
        if not correct and has_label:
            insights.append(
                {
                    "content": sample.rule_text,
                    "section": "domain_concepts",
                    "tags": [sample.concept],
                }
            )
            diagnosis = f"Missed rule for '{sample.concept}'. Captured the correct lesson."
        elif correct:
            diagnosis = "Answer correct; reinforcing the strategy that worked."
        else:
            diagnosis = "Answer wrong but no reliable feedback to learn from."
        return insights, diagnosis
