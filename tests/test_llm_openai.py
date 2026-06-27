"""Tests for the OpenAILLM backend (with a fully mocked client) and the JSON parser."""

import pytest

from ace.llm import _extract_json

openai = pytest.importorskip("openai")


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _Resp:
    def __init__(self, content, usage=(11, 7)):
        self.choices = [_Choice(content)]
        self.usage = _Usage(*usage)


class _Completions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self._content)


class _FakeClient:
    def __init__(self, content, **kwargs):
        self.init_kwargs = kwargs
        self.chat = type("Chat", (), {"completions": _Completions(content)})()


def _patch_openai(monkeypatch, content):
    holder = {}

    def _factory(**kwargs):
        client = _FakeClient(content, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(openai, "OpenAI", _factory)
    return holder


def test_openai_complete_and_usage_accounting(monkeypatch):
    _patch_openai(monkeypatch, "hello from model")
    from ace.llm import OpenAILLM

    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")
    out = llm.complete("system", "user")
    assert out == "hello from model"
    assert llm.num_calls == 1
    assert llm.prompt_tokens == 11 and llm.completion_tokens == 7

    llm.complete("system", "user")
    assert llm.num_calls == 2
    assert llm.prompt_tokens == 22  # accumulates


def test_openai_complete_json_parses_fenced(monkeypatch):
    _patch_openai(monkeypatch, '```json\n{"answer": "B"}\n```')
    from ace.llm import OpenAILLM

    llm = OpenAILLM(api_key="sk-test")
    data = llm.complete_json("system", "user")
    assert data == {"answer": "B"}


def test_openai_passes_model_and_temperature(monkeypatch):
    holder = _patch_openai(monkeypatch, "ok")
    from ace.llm import OpenAILLM

    llm = OpenAILLM(model="gpt-4o", temperature=0.7, api_key="sk-test")
    llm.complete("s", "u")
    kwargs = holder["client"].chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.7
    assert kwargs["messages"][0]["role"] == "system"


# --------------------------------------------------------------------------- #
# JSON extraction additional edge cases
# --------------------------------------------------------------------------- #
def test_extract_json_nested_braces():
    assert _extract_json('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}


def test_extract_json_unterminated_returns_empty():
    assert _extract_json('{"a": 1') == {}
