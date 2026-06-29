"""Recipe 06 — Keep the playbook compact with grow-and-refine.

What you'll learn
-----------------
* How ACE stops the playbook from bloating: **de-duplication** (near-identical
  bullets are merged, counters folded into the survivor) and **pruning**
  (consistently harmful bullets are dropped).
* That you can run :func:`~ace.refine.grow_and_refine` directly on any playbook —
  it's deterministic, non-LLM logic you can call and reason about yourself.

This recipe builds a deliberately messy playbook by hand so you can *see* the
cleanup happen. Runs fully offline — **no API key**.

    python cookbook/06_grow_and_refine.py
"""

from __future__ import annotations

from ace import Bullet, Playbook, grow_and_refine


def build_messy_playbook() -> Playbook:
    pb = Playbook()
    # Two near-duplicate strategies (differ only by a word) — should be merged.
    pb.add(Bullet(content="Always verify identity before cancelling an order.",
                  section="strategies", helpful_count=2))
    pb.add(Bullet(content="Always verify the identity before cancelling an order.",
                  section="strategies", helpful_count=1))
    # A genuinely distinct, useful strategy — should be kept.
    pb.add(Bullet(content="State the arrival estimate explicitly when asked about status.",
                  section="strategies", helpful_count=3))
    # A bullet that has proven harmful far more than helpful — should be pruned.
    pb.add(Bullet(content="Skip identity checks to answer faster.",
                  section="common_mistakes", helpful_count=0, harmful_count=3))
    return pb


def run() -> dict:
    pb = build_messy_playbook()
    before = len(pb)

    result = grow_and_refine(
        pb,
        dedup_threshold=0.8,   # merge bullets above this lexical/semantic similarity
        prune_harmful=True,
        harmful_margin=2,      # drop when harmful_count - helpful_count >= 2
    )

    return {
        "before": before,
        "after": len(pb),
        "deduped": result.deduped,
        "pruned": result.pruned,
        "survivors": [b.content for b in pb],
    }


def main() -> int:
    r = run()
    print(f"Playbook size: {r['before']} → {r['after']} bullets")
    print(f"  merged as duplicates : {len(r['deduped'])}")
    print(f"  pruned as harmful    : {len(r['pruned'])}")
    print("\nSurvivors:")
    for content in r["survivors"]:
        print(f"  - {content}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
