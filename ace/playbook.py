"""Playbook: the evolving, structured context at the heart of ACE.

In ACE, context is *not* a single monolithic prompt. It is a collection of
small, itemized **bullets**, each carrying metadata (a stable id and
helpful/harmful counters) and content (a reusable strategy, domain concept, or
common failure mode). Bullets are organized into named **sections**.

This itemized design is what enables ACE's three core properties:

* **localization** — only the relevant bullets are touched on each update;
* **fine-grained retrieval** — the Generator can focus on pertinent bullets;
* **incremental adaptation** — efficient merge / prune / de-duplication.

See §3.1 of the paper ("Incremental Delta Updates").
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional

# Default sections for a general-purpose playbook. Applications can override
# these via :class:`~ace.config.ACEConfig`.
DEFAULT_SECTIONS: List[str] = [
    "strategies",  # reusable, reproducible procedures
    "domain_concepts",  # facts, definitions, rules of the domain
    "common_mistakes",  # known failure modes / pitfalls to avoid
    "tool_usage",  # how to call tools / APIs correctly
    "formatting",  # output / answer formatting requirements
]


def _new_id() -> str:
    """A short, stable, human-readable bullet id (e.g. ``ctx-1a2b3c4d``)."""
    return f"ctx-{uuid.uuid4().hex[:8]}"


@dataclass
class Bullet:
    """A single itemized unit of context.

    Attributes
    ----------
    id:
        Stable unique identifier. The Generator references these ids when it
        reports which bullets were helpful or harmful, which lets the Curator
        apply *localized* updates instead of rewriting everything.
    content:
        The actual text — a strategy, concept, or pitfall.
    section:
        Which playbook section this bullet belongs to.
    helpful_count / harmful_count:
        Counters incremented from execution feedback. They drive grow-and-refine
        pruning (consistently harmful, never-helpful bullets can be removed).
    tags:
        Optional free-form tags (used by some tasks / demos for grouping).
    created_at_step:
        The adaptation step at which the bullet was first added (for timelines).
    """

    content: str
    section: str = "strategies"
    id: str = field(default_factory=_new_id)
    helpful_count: int = 0
    harmful_count: int = 0
    tags: List[str] = field(default_factory=list)
    created_at_step: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def score(self) -> int:
        """Net usefulness; used for ranking and pruning decisions."""
        return self.helpful_count - self.harmful_count

    def render(self) -> str:
        """How the bullet appears inside the context string given to the model."""
        meta = f"  (helpful={self.helpful_count}, harmful={self.harmful_count})"
        return f"- [{self.id}] {self.content}{meta}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Bullet":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class Playbook:
    """An ordered, sectioned collection of :class:`Bullet` objects.

    The playbook is the "evolving context" that ACE grows and refines over
    time. It is deliberately a plain, inspectable data structure: every change
    is applied by deterministic, non-LLM logic (see :mod:`ace.delta`).
    """

    def __init__(self, sections: Optional[List[str]] = None) -> None:
        self.sections: List[str] = list(sections or DEFAULT_SECTIONS)
        # Insertion-ordered mapping id -> Bullet.
        self._bullets: "Dict[str, Bullet]" = {}

    # ------------------------------------------------------------------ #
    # Basic container behaviour
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._bullets)

    def __iter__(self) -> Iterator[Bullet]:
        return iter(self._bullets.values())

    def __contains__(self, bullet_id: str) -> bool:
        return bullet_id in self._bullets

    @property
    def bullets(self) -> List[Bullet]:
        return list(self._bullets.values())

    def get(self, bullet_id: str) -> Optional[Bullet]:
        return self._bullets.get(bullet_id)

    def section_bullets(self, section: str) -> List[Bullet]:
        return [b for b in self._bullets.values() if b.section == section]

    # ------------------------------------------------------------------ #
    # Mutations — these are the *only* ways the context changes. They are
    # invoked exclusively by deterministic delta application (ace.delta).
    # ------------------------------------------------------------------ #
    def add(self, bullet: Bullet) -> Bullet:
        if bullet.section not in self.sections:
            # Unknown sections are appended rather than rejected, keeping the
            # framework permissive for new domains.
            self.sections.append(bullet.section)
        self._bullets[bullet.id] = bullet
        return bullet

    def update(self, bullet_id: str, content: str) -> bool:
        b = self._bullets.get(bullet_id)
        if b is None:
            return False
        b.content = content
        return True

    def remove(self, bullet_id: str) -> bool:
        return self._bullets.pop(bullet_id, None) is not None

    def mark_helpful(self, bullet_id: str, n: int = 1) -> None:
        b = self._bullets.get(bullet_id)
        if b is not None:
            b.helpful_count += n

    def mark_harmful(self, bullet_id: str, n: int = 1) -> None:
        b = self._bullets.get(bullet_id)
        if b is not None:
            b.harmful_count += n

    # ------------------------------------------------------------------ #
    # Rendering — turn the structured playbook into the context string that
    # gets prepended to the model prompt.
    # ------------------------------------------------------------------ #
    def render(self, include_empty: bool = False) -> str:
        if not self._bullets:
            return "(The playbook is currently empty.)"
        out: List[str] = []
        for section in self.sections:
            items = self.section_bullets(section)
            if not items and not include_empty:
                continue
            title = section.replace("_", " ").title()
            out.append(f"## {title}")
            for b in items:
                out.append(b.render())
            out.append("")
        # Any bullets in sections not in self.sections (defensive).
        rendered_sections = set(self.sections)
        extras = [b for b in self._bullets.values() if b.section not in rendered_sections]
        if extras:
            out.append("## Other")
            out.extend(b.render() for b in extras)
        return "\n".join(out).strip()

    # ------------------------------------------------------------------ #
    # Stats & serialization
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "num_bullets": len(self._bullets),
            "sections": {s: len(self.section_bullets(s)) for s in self.sections},
            "total_helpful": sum(b.helpful_count for b in self._bullets.values()),
            "total_harmful": sum(b.harmful_count for b in self._bullets.values()),
            "approx_tokens": self.approx_tokens(),
        }

    def approx_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) of the rendered context."""
        return max(1, len(self.render()) // 4)

    def to_dict(self) -> dict:
        return {
            "sections": self.sections,
            "bullets": [b.to_dict() for b in self._bullets.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Playbook":
        pb = cls(sections=d.get("sections"))
        for bd in d.get("bullets", []):
            pb.add(Bullet.from_dict(bd))
        return pb

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Playbook":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def clone(self) -> "Playbook":
        return Playbook.from_dict(self.to_dict())
