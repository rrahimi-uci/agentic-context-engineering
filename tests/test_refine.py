from ace.playbook import Bullet, Playbook
from ace.refine import grow_and_refine, _jaccard


def test_jaccard_identical():
    assert _jaccard("the quick brown fox", "the quick brown fox") == 1.0


def test_jaccard_disjoint():
    assert _jaccard("apple banana", "carrot date") == 0.0


def test_dedup_removes_near_duplicates():
    pb = Playbook()
    pb.add(Bullet(content="Convert virtual currency to USD before closing", section="domain_concepts"))
    pb.add(Bullet(content="Convert virtual currency to USD before closing", section="domain_concepts"))
    result = grow_and_refine(pb, dedup_threshold=0.8, prune_harmful=False)
    assert len(pb) == 1
    assert len(result.deduped) == 1


def test_dedup_folds_counters():
    pb = Playbook()
    a = pb.add(Bullet(content="same text here always", section="strategies", helpful_count=2))
    b = pb.add(Bullet(content="same text here always", section="strategies", helpful_count=3))
    grow_and_refine(pb, dedup_threshold=0.8, prune_harmful=False)
    # Survivor keeps folded counters.
    survivor = pb.bullets[0]
    assert survivor.helpful_count == 5


def test_prune_harmful():
    pb = Playbook()
    b = pb.add(Bullet(content="bad advice", harmful_count=3, helpful_count=0))
    result = grow_and_refine(pb, prune_harmful=True, harmful_margin=2)
    assert len(pb) == 0
    assert b.id in result.pruned


def test_keeps_distinct_bullets():
    pb = Playbook()
    pb.add(Bullet(content="completely different topic one", section="strategies"))
    pb.add(Bullet(content="another unrelated subject entirely", section="strategies"))
    grow_and_refine(pb, dedup_threshold=0.86, prune_harmful=False)
    assert len(pb) == 2


def test_embedder_path():
    # A trivial embedder: bag-of-words counts over a tiny vocab.
    vocab = {}

    def embed(texts):
        for t in texts:
            for w in t.lower().split():
                vocab.setdefault(w, len(vocab))
        out = []
        for t in texts:
            v = [0.0] * (len(vocab) + 1)
            for w in t.lower().split():
                v[vocab[w]] += 1
            out.append(v)
        return out

    pb = Playbook()
    pb.add(Bullet(content="alpha beta gamma", section="strategies"))
    pb.add(Bullet(content="alpha beta gamma", section="strategies"))
    grow_and_refine(pb, dedup_threshold=0.99, prune_harmful=False, embedder=embed)
    assert len(pb) == 1
