"""Grow-and-refine (§3.2).

Beyond incremental growth, ACE keeps the playbook compact and relevant through
periodic or lazy refinement:

* bullets with new ids are **appended**;
* existing bullets are **updated in place** (e.g. counter increments — handled
  during delta merge);
* a **de-duplication** step prunes redundancy by comparing bullets via semantic
  embeddings;
* optionally, consistently **harmful / never-helpful** bullets are pruned.

Refinement can run *proactively* (after every delta) or *lazily* (only when the
context exceeds a size budget).

Embeddings are optional. If an embedder is provided (e.g. OpenAI embeddings) we
use cosine similarity; otherwise we fall back to a dependency-free lexical
(token-overlap) similarity so the framework works fully offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .playbook import Playbook

Embedder = Callable[[Sequence[str]], "Sequence[Sequence[float]]"]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    import numpy as np

    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


@dataclass
class RefineResult:
    deduped: List[str]  # ids removed as duplicates
    pruned: List[str]  # ids removed for being harmful / unhelpful

    @property
    def num_removed(self) -> int:
        return len(self.deduped) + len(self.pruned)


def grow_and_refine(
    playbook: Playbook,
    *,
    dedup_threshold: float = 0.86,
    prune_harmful: bool = True,
    harmful_margin: int = 2,
    embedder: Optional[Embedder] = None,
) -> RefineResult:
    """De-duplicate and (optionally) prune a playbook in place.

    Parameters
    ----------
    dedup_threshold:
        Similarity above which two bullets in the same section are considered
        duplicates. The newer (later) bullet is dropped; its counters are folded
        into the survivor.
    prune_harmful:
        If True, remove bullets whose ``harmful_count`` exceeds ``helpful_count``
        by at least ``harmful_margin``.
    embedder:
        Optional batched embedding function. Falls back to lexical similarity.
    """
    deduped: List[str] = []
    pruned: List[str] = []

    # --- de-duplication, section by section --- #
    for section in list(playbook.sections):
        bullets = playbook.section_bullets(section)
        if len(bullets) < 2:
            continue

        vectors = None
        if embedder is not None:
            try:
                vectors = list(embedder([b.content for b in bullets]))
            except Exception:
                vectors = None

        survivors: List[int] = []
        for i, b in enumerate(bullets):
            dup_of = None
            for j in survivors:
                if vectors is not None:
                    sim = _cosine(vectors[i], vectors[j])
                else:
                    sim = _jaccard(b.content, bullets[j].content)
                if sim >= dedup_threshold:
                    dup_of = j
                    break
            if dup_of is None:
                survivors.append(i)
            else:
                # Fold counters into the survivor, then drop the duplicate.
                keep = bullets[dup_of]
                keep.helpful_count += b.helpful_count
                keep.harmful_count += b.harmful_count
                playbook.remove(b.id)
                deduped.append(b.id)

    # --- prune consistently harmful bullets --- #
    if prune_harmful:
        for b in list(playbook):
            if b.harmful_count - b.helpful_count >= harmful_margin:
                playbook.remove(b.id)
                pruned.append(b.id)

    return RefineResult(deduped=deduped, pruned=pruned)


def make_openai_embedder(
    model: str = "text-embedding-3-small", api_key: Optional[str] = None
) -> Embedder:
    """Build an OpenAI-backed batched embedder for semantic de-duplication."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def _embed(texts: Sequence[str]) -> List[List[float]]:
        resp = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]

    return _embed
