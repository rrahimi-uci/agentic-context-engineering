import pytest

from ace import ACE, ACEConfig, SimulatedLLM, TeachingEnvironment, build_teaching_task
from ace.baselines import MonolithicRewriteAgent, StaticAgent


@pytest.fixture
def env():
    return TeachingEnvironment(known_fraction=0.3, seed=2)


@pytest.fixture
def task():
    return build_teaching_task(repeats=3, seed=2)


def test_base_below_ace_offline(env, task):
    train, test = task.split(train_frac=0.5, seed=2)
    base = StaticAgent(SimulatedLLM(env)).run(test)
    ace = ACE(SimulatedLLM(env), ACEConfig(epochs=4))
    ace.adapt_offline(train)
    res = ace.evaluate(test)
    assert res.accuracy > base.accuracy
    assert len(ace.playbook) > 0


def test_online_improves_over_base(env, task):
    base = StaticAgent(SimulatedLLM(env)).run(task)
    ace = ACE(SimulatedLLM(env))
    online = ace.adapt_online(task)
    assert online.accuracy >= base.accuracy


def test_offline_warmup_helps(env, task):
    train, test = task.split(train_frac=0.4, seed=2)
    cold = ACE(SimulatedLLM(env)).adapt_online(test).accuracy
    warm = ACE(SimulatedLLM(env), ACEConfig(epochs=5))
    warm.adapt_offline(train)
    warm_acc = warm.adapt_online(test).accuracy
    assert warm_acc >= cold


def test_monolithic_collapses(env, task):
    mono = MonolithicRewriteAgent(SimulatedLLM(env), collapse_prob=0.3).run(task)
    collapses = sum(1 for r in mono.history if r.refine.get("collapsed"))
    assert collapses > 0


def test_ace_beats_monolithic(env, task):
    mono = MonolithicRewriteAgent(SimulatedLLM(env), collapse_prob=0.25).run(task)
    ace = ACE(SimulatedLLM(env)).adapt_online(task)
    assert ace.accuracy >= mono.accuracy


def test_step_records_are_complete(env, task):
    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task)
    assert len(result.history) == len(task.samples)
    r = result.history[0]
    assert r.step == 1
    assert r.playbook_size >= 0
    assert "added" in r.merge


def test_callback_receives_records(env, task):
    seen = []
    ace = ACE(SimulatedLLM(env))
    ace.adapt_online(task, callback=lambda rec: seen.append(rec))
    assert len(seen) == len(task.samples)


def test_label_free_setting(env, task):
    # Without labels and without execution signal, ACE should not crash and
    # should not magically learn rules it can't infer.
    ace = ACE(SimulatedLLM(env), ACEConfig(use_labels=False))
    result = ace.adapt_online(task)
    assert result.summary()["steps"] == len(task.samples)


def test_runresult_metrics(env, task):
    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task)
    assert 0 <= result.accuracy <= 100
    assert len(result.growth_curve) == len(task.samples)
    assert len(result.token_curve) == len(task.samples)


def test_grow_and_refine_keeps_playbook_bounded(env):
    # Many repeats of the same concepts should be de-duplicated, not unbounded.
    task = build_teaching_task(repeats=8, seed=2)
    ace = ACE(SimulatedLLM(env), ACEConfig(refine_every=1))
    ace.adapt_online(task)
    # At most one bullet per distinct concept (12 in the rule bank).
    assert len(ace.playbook) <= 12


def test_predict_does_not_mutate(env, task):
    ace = ACE(SimulatedLLM(env))
    before = len(ace.playbook)
    ace.predict(task.samples[0])
    assert len(ace.playbook) == before
