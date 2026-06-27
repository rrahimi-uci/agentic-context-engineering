"""Tests for grow-and-refine: embedding-based dedup, counter folding, pruning."""

from ace.playbook import Bullet, Playbook
from ace.refine import grow_and_refine


def _fake_embedder(mapping):
    """Return a batched embedder that maps known texts to fixed vectors."""

    def _embed(texts):
        return [mapping[t] for t in texts]

    return _embed


def test_embedder_dedup_folds_counters():
    pb = Playbook()
    a = pb.add(Bullet(content="alpha", section="strategies"))
    b = pb.add(Bullet(content="beta", section="strategies"))  # semantically identical vector
    pb.mark_helpful(a.id, 1)
    pb.mark_helpful(b.id, 2)
    pb.mark_harmful(b.id, 1)

    embedder = _fake_embedder({"alpha": [1.0, 0.0], "beta": [1.0, 0.0]})
    res = grow_and_refine(pb, dedup_threshold=0.9, prune_harmful=False, embedder=embedder)

    assert b.id in res.deduped
    assert b.id not in pb and a.id in pb
    survivor = pb.get(a.id)
    # Survivor absorbs the duplicate's counters.
    assert survivor.helpful_count == 3
    assert survivor.harmful_count == 1


def test_embedder_keeps_distinct_bullets():
    pb = Playbook()
    pb.add(Bullet(content="alpha", section="strategies"))
    pb.add(Bullet(content="gamma", section="strategies"))
    embedder = _fake_embedder({"alpha": [1.0, 0.0], "gamma": [0.0, 1.0]})
    res = grow_and_refine(pb, dedup_threshold=0.9, prune_harmful=False, embedder=embedder)
    assert res.deduped == []
    assert len(pb) == 2


def test_embedder_exception_falls_back_to_lexical():
    pb = Playbook()
    pb.add(Bullet(content="always verify the units", section="strategies"))
    pb.add(Bullet(content="always verify the units", section="strategies"))  # identical text

    def boom(_texts):
        raise RuntimeError("embedding service down")

    res = grow_and_refine(pb, dedup_threshold=0.9, prune_harmful=False, embedder=boom)
    # Lexical Jaccard on identical text == 1.0 -> deduped despite embedder failure.
    assert len(res.deduped) == 1
    assert len(pb) == 1


def test_prune_harmful_bullets():
    pb = Playbook()
    good = pb.add(Bullet(content="keep me", section="strategies"))
    bad = pb.add(Bullet(content="drop me", section="strategies"))
    pb.mark_harmful(bad.id, 3)  # harmful - helpful = 3 >= margin(2)
    res = grow_and_refine(pb, dedup_threshold=0.99, prune_harmful=True, harmful_margin=2)
    assert bad.id in res.pruned
    assert good.id in pb and bad.id not in pb


def test_lexical_dedup_without_embedder():
    pb = Playbook()
    pb.add(Bullet(content="parse the date in ISO 8601 format", section="formatting"))
    pb.add(Bullet(content="parse the date in ISO 8601 format", section="formatting"))
    res = grow_and_refine(pb, dedup_threshold=0.85, prune_harmful=False)
    assert len(res.deduped) == 1
