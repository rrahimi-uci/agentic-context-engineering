"""Edge-case tests for Feedback and deterministic delta merging."""

from ace.delta import DeltaContext, DeltaOp, DeltaOperation, apply_delta
from ace.feedback import Feedback
from ace.playbook import Bullet, Playbook


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
def test_feedback_has_label():
    assert Feedback(correct=True).has_label is True
    assert Feedback(ground_truth="x").has_label is True
    assert Feedback(signal="only execution signal").has_label is False
    assert Feedback().has_label is False


# --------------------------------------------------------------------------- #
# DeltaOperation parsing
# --------------------------------------------------------------------------- #
def test_delta_operation_from_dict_normalizes_op_and_id_alias():
    op = DeltaOperation.from_dict({"op": "update", "id": "ctx-1", "content": "c"})
    assert op.op is DeltaOp.UPDATE
    assert op.target_id == "ctx-1"  # 'id' is accepted as an alias for target_id


def test_delta_operation_roundtrip():
    op = DeltaOperation(op=DeltaOp.ADD, section="strategies", content="c", tags=["t"])
    again = DeltaOperation.from_dict(op.to_dict())
    assert again.op is DeltaOp.ADD and again.content == "c" and again.tags == ["t"]


def test_delta_context_is_empty():
    assert DeltaContext().is_empty() is True
    assert DeltaContext(helpful_ids=["ctx-1"]).is_empty() is False


# --------------------------------------------------------------------------- #
# apply_delta
# --------------------------------------------------------------------------- #
def test_apply_delta_add_update_remove():
    pb = Playbook()
    add = DeltaOperation(op=DeltaOp.ADD, section="strategies", content="new rule")
    res = apply_delta(pb, DeltaContext(operations=[add]), step=5)
    assert len(res.added) == 1
    new_id = res.added[0]
    assert pb.get(new_id).created_at_step == 5

    upd = DeltaOperation(op=DeltaOp.UPDATE, target_id=new_id, content="sharper rule")
    res2 = apply_delta(pb, DeltaContext(operations=[upd]))
    assert res2.updated == [new_id]
    assert pb.get(new_id).content == "sharper rule"

    rem = DeltaOperation(op=DeltaOp.REMOVE, target_id=new_id)
    res3 = apply_delta(pb, DeltaContext(operations=[rem]))
    assert res3.removed == [new_id]
    assert new_id not in pb


def test_apply_delta_skips_empty_add_and_missing_targets():
    pb = Playbook()
    ops = [
        DeltaOperation(op=DeltaOp.ADD, content="   "),  # empty -> skipped
        DeltaOperation(op=DeltaOp.UPDATE, target_id="nope", content="x"),  # no such id
        DeltaOperation(op=DeltaOp.REMOVE, target_id="nope"),  # no such id
    ]
    res = apply_delta(pb, DeltaContext(operations=ops))
    assert res.added == [] and res.updated == [] and res.removed == []
    assert len(pb) == 0


def test_apply_delta_marks_helpful_harmful_only_for_existing():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    delta = DeltaContext(helpful_ids=[b.id, "ghost"], harmful_ids=[b.id])
    res = apply_delta(pb, delta)
    assert res.helpful_marked == 1  # 'ghost' ignored
    assert res.harmful_marked == 1
    assert pb.get(b.id).helpful_count == 1
    assert pb.get(b.id).harmful_count == 1


def test_merge_result_num_changes():
    pb = Playbook()
    ops = [DeltaOperation(op=DeltaOp.ADD, content="a"), DeltaOperation(op=DeltaOp.ADD, content="b")]
    res = apply_delta(pb, DeltaContext(operations=ops))
    assert res.num_changes == 2
