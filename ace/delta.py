"""Incremental delta updates (§3.1).

Instead of asking an LLM to rewrite the whole context on every step — the
practice that causes **context collapse** (Figure 2 in the paper) — ACE
represents each update as a small set of *delta operations* over individual
bullets. The Curator proposes these operations; they are then merged into the
playbook by the **deterministic, non-LLM logic** in this module.

Because operations are itemized and localized, many deltas can be merged in
parallel, and accumulated knowledge can never be silently erased by a runaway
summarization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .playbook import Bullet, Playbook


class DeltaOp(str, Enum):
    ADD = "ADD"  # introduce a brand-new bullet
    UPDATE = "UPDATE"  # edit the content of an existing bullet
    REMOVE = "REMOVE"  # delete a bullet (e.g. obsolete / wrong)


@dataclass
class DeltaOperation:
    """A single localized edit to the playbook."""

    op: DeltaOp
    section: str = "strategies"
    content: str = ""
    target_id: Optional[str] = None  # required for UPDATE / REMOVE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DeltaOperation":
        op = DeltaOp(str(d.get("op", "ADD")).upper())
        return cls(
            op=op,
            section=d.get("section", "strategies"),
            content=d.get("content", ""),
            target_id=d.get("target_id") or d.get("id"),
            tags=list(d.get("tags", [])),
            metadata=dict(d.get("metadata", {})),
        )

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "section": self.section,
            "content": self.content,
            "target_id": self.target_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class DeltaContext:
    """A compact batch of operations produced for one adaptation step.

    Also carries the *usage feedback* surfaced by the Generator: which existing
    bullets were helpful or harmful while solving the query. That feedback is
    applied in place (counter increments) — the grow-and-refine "update existing
    bullets" path.
    """

    operations: List[DeltaOperation] = field(default_factory=list)
    helpful_ids: List[str] = field(default_factory=list)
    harmful_ids: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.operations and not self.helpful_ids and not self.harmful_ids

    def to_dict(self) -> dict:
        return {
            "operations": [o.to_dict() for o in self.operations],
            "helpful_ids": self.helpful_ids,
            "harmful_ids": self.harmful_ids,
        }


@dataclass
class MergeResult:
    """Summary of what changed when a :class:`DeltaContext` was applied."""

    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    helpful_marked: int = 0
    harmful_marked: int = 0

    @property
    def num_changes(self) -> int:
        return len(self.added) + len(self.updated) + len(self.removed)


def apply_delta(playbook: Playbook, delta: DeltaContext, step: int = 0) -> MergeResult:
    """Deterministically merge a delta into a playbook.

    This function contains **no LLM calls** — it is plain, auditable logic. That
    is precisely what protects ACE from context collapse: the model proposes
    small edits, but the merge is mechanical and never rewrites the whole thing.
    """
    result = MergeResult()

    # 1. Apply usage feedback (in-place counter updates).
    for bid in delta.helpful_ids:
        if bid in playbook:
            playbook.mark_helpful(bid)
            result.helpful_marked += 1
    for bid in delta.harmful_ids:
        if bid in playbook:
            playbook.mark_harmful(bid)
            result.harmful_marked += 1

    # 2. Apply structural operations.
    for opn in delta.operations:
        if opn.op is DeltaOp.ADD:
            if not opn.content.strip():
                continue
            bullet = Bullet(
                content=opn.content.strip(),
                section=opn.section or "strategies",
                tags=list(opn.tags),
                created_at_step=step,
                metadata=dict(opn.metadata),
            )
            playbook.add(bullet)
            result.added.append(bullet.id)
        elif opn.op is DeltaOp.UPDATE:
            if opn.target_id and playbook.update(opn.target_id, opn.content.strip()):
                result.updated.append(opn.target_id)
        elif opn.op is DeltaOp.REMOVE:
            if opn.target_id and playbook.remove(opn.target_id):
                result.removed.append(opn.target_id)

    return result
