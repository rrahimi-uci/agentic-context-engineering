from ace.tasks import build_teaching_task, TeachingEnvironment, Sample
from ace.playbook import Bullet, Playbook


def test_teaching_task_builds():
    task = build_teaching_task(repeats=2, seed=0)
    assert len(task.samples) == 24  # 12 rules * 2 repeats
    s = task.samples[0]
    assert s.concept and s.rule_text and s.answer


def test_evaluate_exact_and_text():
    task = build_teaching_task(repeats=1, seed=0)
    s = task.samples[0]
    assert task.evaluate(s.answer, s) is True
    assert task.evaluate("totally wrong", s) is False


def test_split_is_disjoint():
    task = build_teaching_task(repeats=2, seed=0)
    train, test = task.split(train_frac=0.5, seed=0)
    train_ids = {s.id for s in train.samples}
    test_ids = {s.id for s in test.samples}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(task.samples)


def test_env_knows_is_deterministic():
    e1 = TeachingEnvironment(seed=42)
    e2 = TeachingEnvironment(seed=42)
    concepts = ["a", "b", "c", "d", "reserves_multi_property"]
    assert [e1.knows(c) for c in concepts] == [e2.knows(c) for c in concepts]


def test_env_generate_uses_playbook():
    env = TeachingEnvironment(known_fraction=0.0, seed=1)  # base knows nothing
    task = build_teaching_task(repeats=1, seed=1)
    s = task.samples[0]
    pb = Playbook()
    ans_before, *_ = env.generate(s, pb)
    assert task.evaluate(ans_before, s) is False  # can't answer without rule
    pb.add(Bullet(content=s.rule_text, section="domain_concepts", tags=[s.concept]))
    ans_after, _r, used, helpful, _h = env.generate(s, pb)
    assert task.evaluate(ans_after, s) is True
    assert helpful  # the bullet was credited


def test_reflect_requires_label():
    env = TeachingEnvironment(seed=1)
    s = build_teaching_task(repeats=1, seed=1).samples[0]
    insights, _ = env.reflect(s, correct=False, has_label=True)
    assert insights and insights[0]["tags"] == [s.concept]
    none_insights, _ = env.reflect(s, correct=False, has_label=False)
    assert none_insights == []


def test_openai_agents_instructions_builder():
    # The dynamic-instructions builder should not require the SDK to construct.
    from ace import ACE
    from ace.llm import SimulatedLLM
    from ace.integrations.openai_agents import playbook_instructions

    env = TeachingEnvironment(seed=1)
    ace = ACE(SimulatedLLM(env))
    ace.playbook.add(Bullet(content="be concise"))
    fn = playbook_instructions("You are helpful.", ace)
    text = fn(None, None)
    assert "You are helpful." in text
    assert "be concise" in text
    assert "Playbook" in text
