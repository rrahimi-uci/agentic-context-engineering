"""Token / cost usage surfacing through StepRecord and RunResult."""

from ace import (
    ACE,
    ACEConfig,
    Feedback,
    Sample,
    SimulatedLLM,
    TeachingEnvironment,
    build_teaching_task,
)


class _CountingLLM:
    """A minimal real-style backend that returns canned JSON and counts usage."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.num_calls = 0

    def _tick(self):
        self.num_calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 3
        self.cached_prompt_tokens += 2

    def complete(self, system, user, **kwargs):
        self._tick()
        return "answer"

    def complete_json(self, system, user, **kwargs):
        self._tick()
        return {
            "answer": "x",
            "insights": [{"content": "a reusable lesson", "section": "strategies"}],
            "operations": [{"op": "ADD", "section": "strategies", "content": "a reusable lesson"}],
            "helpful_ids": [],
            "harmful_ids": [],
        }


def test_step_record_captures_token_usage():
    llm = _CountingLLM()
    ace = ACE(llm, ACEConfig(reflector_max_rounds=1))
    rec = ace.step(Sample(id="1", question="q?"), Feedback(signal="env: ok"))
    # generate + reflect + curate = 3 model calls within step().
    assert rec.llm_calls == 3
    assert rec.prompt_tokens == 30
    assert rec.completion_tokens == 9
    assert rec.cached_prompt_tokens == 6


def test_shared_backend_counted_once_per_call():
    # All three roles share one backend instance; usage must not be tripled.
    llm = _CountingLLM()
    ace = ACE(llm, ACEConfig(reflector_max_rounds=1))
    rec = ace.step(Sample(id="1", question="q?"), Feedback(signal="env"))
    assert rec.llm_calls == llm.num_calls == 3


def test_runresult_aggregates_and_summarizes_usage():
    env = TeachingEnvironment(seed=1)
    task = build_teaching_task(repeats=1, seed=1)
    ace = ACE(SimulatedLLM(env))
    result = ace.adapt_online(task)

    s = result.summary()
    for key in (
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
    ):
        assert key in s

    # Totals equal the sum over per-step records.
    assert result.total_llm_calls == sum(r.llm_calls for r in result.history)
    assert s["total_tokens"] == result.total_prompt_tokens + result.total_completion_tokens
    # SimulatedLLM reports no tokens, but call accounting is wired end-to-end:
    # each online step = generate + reflect + curate = 3 calls.
    assert result.total_llm_calls > 0
    assert all(r.llm_calls == 3 for r in result.history)


def test_split_usage_preserves_total():
    total = {"prompt_tokens": 10, "completion_tokens": 7, "cached_prompt_tokens": 0, "llm_calls": 4}
    parts = ACE._split_usage(total, 3)
    assert len(parts) == 3
    for key, value in total.items():
        assert sum(p[key] for p in parts) == value  # nothing lost or invented
    assert ACE._split_usage(total, 0) == []


def test_parallel_evaluate_matches_sequential():
    env = TeachingEnvironment(seed=3)
    task = build_teaching_task(repeats=2, seed=3)
    seq = ACE(SimulatedLLM(env)).evaluate(task)
    par = ACE(SimulatedLLM(env)).evaluate(task, max_workers=4)
    # Inference-only: results are identical regardless of concurrency.
    assert par.accuracy == seq.accuracy
    assert [r.prediction for r in par.history] == [r.prediction for r in seq.history]
    assert len(par.history) == len(task.samples)


def test_simulated_backend_has_no_auto_embedder():
    # Auto-wiring only triggers for an OpenAILLM backend; lexical dedup otherwise.
    ace = ACE(SimulatedLLM(TeachingEnvironment(seed=1)))
    assert ace.embedder is None
