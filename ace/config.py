"""Configuration for the ACE engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .playbook import DEFAULT_SECTIONS


@dataclass
class ACEConfig:
    """Knobs for the ACE adaptation loop.

    Defaults follow the paper where applicable: up to 5 Reflector refinement
    rounds and up to 5 epochs of offline adaptation, batch size 1.
    """

    # Roles / refinement.
    reflector_max_rounds: int = 5
    curator_use_llm: bool = True  # if False, Curator builds deltas deterministically
    sections: List[str] = field(default_factory=lambda: list(DEFAULT_SECTIONS))

    # Offline adaptation.
    epochs: int = 5

    # Grow-and-refine.
    refine_every: int = 1  # proactive: refine after every N deltas (0 = lazy only)
    dedup_threshold: float = 0.86
    prune_harmful: bool = True
    harmful_margin: int = 2
    lazy_refine_token_budget: Optional[int] = None  # lazy: refine when tokens exceed this

    # Behaviour.
    use_labels: bool = True  # whether ground-truth labels are given to the Reflector
    verbose: bool = False
