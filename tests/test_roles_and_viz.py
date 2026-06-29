"""Tests for roles with a fake JSON LLM, the LLM JSON parser, and visualization."""

from ace.engine import ACE, RunResult, StepRecord
from ace.feedback import Feedback
from ace.llm import _extract_json
from ace.playbook import Playbook
from ace.roles import Curator, Generator, Reflector
from ace.tasks import Sample
from ace.visualize import _windowed_accuracy, render_html_report


class FakeLLM:
    """A scripted LLM that returns canned JSON for each role."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, **kw):
        return "ok"

    def complete_json(self, system, user, **kw):
        self.calls += 1
        if "You are the Generator" in system:
            return {"answer": "B", "reasoning": "because", "helpful_ids": [], "harmful_ids": []}
        if "You are the Reflector" in system:
            return {
                "diagnosis": "wrong",
                "insights": [
                    {
                        "content": "always verify units",
                        "section": "common_mistakes",
                        "tags": ["units"],
                    }
                ],
                "helpful_ids": [],
                "harmful_ids": [],
            }
        return {"operations": []}


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert _extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_extract_json_embedded():
    assert _extract_json('blah blah {"a": 3} trailing') == {"a": 3}


def test_extract_json_garbage():
    assert _extract_json("no json here") == {}


def test_generator_with_fake_llm():
    gen = Generator(FakeLLM())
    g = gen.generate(Sample(id="1", question="q?"), Playbook())
    assert g.answer == "B"


def test_reflector_produces_insights():
    refl = Reflector(FakeLLM(), max_rounds=1)
    g = Generator(FakeLLM()).generate(Sample(id="1", question="q?"), Playbook())
    r = refl.reflect(
        Sample(id="1", question="q?"), g, Feedback(correct=False, ground_truth="A"), Playbook()
    )
    assert r.insights and r.insights[0]["content"] == "always verify units"


def test_curator_builds_add_ops():
    refl = Reflector(FakeLLM(), max_rounds=1)
    sample = Sample(id="1", question="q?")
    g = Generator(FakeLLM()).generate(sample, Playbook())
    r = refl.reflect(sample, g, Feedback(correct=False, ground_truth="A"), Playbook())
    delta = Curator(FakeLLM()).curate(sample, g, r, Playbook())
    assert len(delta.operations) == 1
    assert delta.operations[0].content == "always verify units"


def test_full_loop_with_fake_llm():
    ace = ACE(FakeLLM())
    sample = Sample(id="1", question="q?", answer="A")
    rec = ace.step(sample, Feedback(correct=False, ground_truth="A"))
    assert rec.playbook_size == 1  # one insight added


def test_windowed_accuracy():
    recs = [
        StepRecord(
            step=i,
            phase="x",
            sample_id=str(i),
            question="q",
            prediction="",
            ground_truth="",
            correct=(i % 2 == 0),
            delta={},
            merge={},
            refine={},
            playbook_size=i,
            playbook_tokens=i,
        )
        for i in range(1, 7)
    ]
    acc = _windowed_accuracy(recs, window=3)
    assert len(acc) == 6
    assert all(0 <= a <= 100 for a in acc)


def test_render_html_report_is_self_contained():
    recs = [
        StepRecord(
            step=i,
            phase="online",
            sample_id=str(i),
            question="q",
            prediction="p",
            ground_truth="g",
            correct=(i > 2),
            delta={"operations": [{"content": "x"}]},
            merge={"added": ["ctx-1"]},
            refine={},
            playbook_size=i,
            playbook_tokens=i * 10,
        )
        for i in range(1, 6)
    ]
    pb = Playbook()
    from ace.playbook import Bullet

    pb.add(Bullet(content="learned rule"))
    rr = RunResult(history=recs, playbook=pb)
    html = render_html_report({"ACE": rr})
    assert "<!DOCTYPE html>" in html
    assert "learned rule" in html
    assert "<svg" in html
