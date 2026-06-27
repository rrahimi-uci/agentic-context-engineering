"""Feedback signals consumed by the Reflector.

ACE works in two regimes (see Tables 1 & 2 of the paper):

* **with ground-truth labels** — the Reflector is told whether the answer was
  correct and what the correct answer was;
* **without labels** — the Reflector relies purely on *natural execution
  feedback*, e.g. "the code raised a TypeError" or "the API returned 404".

Both are represented by a single :class:`Feedback` object so the rest of the
framework does not care which regime it is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Feedback:
    """A unit of feedback about one generation.

    Attributes
    ----------
    correct:
        Whether the answer was correct, if known. ``None`` means "no label".
    ground_truth:
        The reference answer, if available (offline / labeled setting).
    signal:
        Free-form natural-language execution feedback (errors, env responses,
        validator messages). This is what powers the *label-free* setting.
    """

    correct: Optional[bool] = None
    ground_truth: Optional[str] = None
    signal: str = ""

    @property
    def has_label(self) -> bool:
        return self.correct is not None or self.ground_truth is not None
