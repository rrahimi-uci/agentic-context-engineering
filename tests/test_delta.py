from ace.delta import DeltaContext, DeltaOp, DeltaOperation, apply_delta
from ace.playbook import Bullet, Playbook


def test_add_operation():
    pb = Playbook()
    delta = DeltaContext(operations=[
        DeltaOperation(op=DeltaOp.ADD, section="strategies", content="new strat"),
    ])
    result = apply_delta(pb, delta, step=1)
    assert len(pb) == 1
    assert len(result.added) == 1
    assert pb.bullets[0].created_at_step == 1


def test_empty_add_is_ignored():
    pb = Playbook()
    apply_delta(pb, DeltaContext(operations=[DeltaOperation(op=DeltaOp.ADD, content="   ")]))
    assert len(pb) == 0


def test_update_operation():
    pb = Playbook()
    b = pb.add(Bullet(content="old"))
    delta = DeltaContext(operations=[
        DeltaOperation(op=DeltaOp.UPDATE, target_id=b.id, content="updated"),
    ])
    result = apply_delta(pb, delta)
    assert pb.get(b.id).content == "updated"
    assert result.updated == [b.id]


def test_remove_operation():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    result = apply_delta(pb, DeltaContext(operations=[
        DeltaOperation(op=DeltaOp.REMOVE, target_id=b.id),
    ]))
    assert len(pb) == 0
    assert result.removed == [b.id]


def test_usage_feedback_increments_counters():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    delta = DeltaContext(helpful_ids=[b.id], harmful_ids=[b.id])
    result = apply_delta(pb, delta)
    assert pb.get(b.id).helpful_count == 1
    assert pb.get(b.id).harmful_count == 1
    assert result.helpful_marked == 1 and result.harmful_marked == 1


def test_feedback_on_missing_id_is_noop():
    pb = Playbook()
    result = apply_delta(pb, DeltaContext(helpful_ids=["ghost"]))
    assert result.helpful_marked == 0


def test_delta_from_dict():
    op = DeltaOperation.from_dict({"op": "add", "content": "c", "id": "ctx-1"})
    assert op.op is DeltaOp.ADD
    assert op.target_id == "ctx-1"


def test_is_empty():
    assert DeltaContext().is_empty()
    assert not DeltaContext(helpful_ids=["a"]).is_empty()
