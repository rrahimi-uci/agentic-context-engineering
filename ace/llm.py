"""LLM backends used by the Generator, Reflector, and Curator.

ACE is model-agnostic. The framework only needs two operations from a backend:

* ``complete(system, user)`` — return free-form text;
* ``complete_json(system, user, schema_hint)`` — return a parsed JSON object.

Two backends ship in the box:

* :class:`OpenAILLM` — talks to the real OpenAI API (chat completions). Use this
  for genuine benchmarks and production agents.
* :class:`SimulatedLLM` — a deterministic, offline backend used by the test
  suite and the bundled demos so that **everything runs with zero API keys and
  is fully reproducible**. It is intentionally simple but faithfully exercises
  the ACE control loop (generate → reflect → curate → merge).

The simulated backend is wired to :mod:`ace.tasks`' teaching environment, where
each question is associated with a hidden domain rule. This lets the demos show
*real, measured* accuracy gains as ACE accumulates the right bullets — without
pretending those numbers come from a frontier model.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Minimal interface every backend must satisfy."""

    def complete(self, system: str, user: str, **kwargs) -> str: ...

    def complete_json(self, system: str, user: str, **kwargs) -> dict: ...


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    # Strip ```json fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


class OpenAILLM:
    """OpenAI Chat Completions backend.

    Uses the Chat Completions API — supported by every current OpenAI chat model
    and by OpenAI-compatible providers (set ``base_url`` to e.g. a vLLM/Together
    endpoint). The official client's exponential backoff is enabled via
    ``max_retries``, so transient 429/5xx/connection errors don't fail a run.

    Parameters
    ----------
    model:
        Any chat model id (default ``gpt-4o-mini``). Pass a newer model
        (e.g. ``gpt-4.1-mini``) as it becomes available — ACE uses the same model
        for all three roles to isolate the benefit of context.
    api_key:
        Falls back to the ``OPENAI_API_KEY`` environment variable.
    temperature:
        Sampling temperature shared by all three roles.
    base_url:
        Optional OpenAI-compatible endpoint.
    max_retries:
        Automatic retries (with exponential backoff) on transient errors,
        handled by the OpenAI client.
    timeout:
        Per-request timeout in seconds.
    json_mode:
        When True (default), :meth:`complete_json` requests guaranteed JSON via
        ``response_format={"type": "json_object"}``. If the endpoint rejects it
        (some compatible providers do), it transparently retries once without it
        and falls back to robust text extraction.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        json_mode: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OpenAILLM requires the 'openai' package. Install with "
                "`pip install ace-playbook[openai]`."
            ) from exc
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        # Lightweight usage accounting (summed across all calls). ``cached_prompt_tokens``
        # is the slice of prompt tokens served from OpenAI's automatic prefix cache —
        # the static system + playbook prefix is cached for you (no code required), so a
        # rising cached share is the visible payoff of ACE's stable-prefix prompts.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.num_calls = 0

    def _chat(
        self, system: str, user: str, *, response_format: Optional[dict] = None, **kwargs
    ) -> str:
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**create_kwargs)
        self.num_calls += 1
        if resp.usage:
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
            details = getattr(resp.usage, "prompt_tokens_details", None)
            if details is not None:
                self.cached_prompt_tokens += getattr(details, "cached_tokens", 0) or 0
        return resp.choices[0].message.content or ""

    def complete(self, system: str, user: str, **kwargs) -> str:
        return self._chat(system, user, **kwargs)

    def complete_json(self, system: str, user: str, **kwargs) -> dict:
        system_json = system + "\n\nRespond with a single valid JSON object and nothing else."
        response_format = {"type": "json_object"} if self.json_mode else None
        try:
            raw = self._chat(system_json, user, response_format=response_format, **kwargs)
        except Exception:
            if response_format is None:
                raise
            # Endpoint rejected json_object mode — retry once as plain text.
            raw = self._chat(system_json, user, response_format=None, **kwargs)
        return _extract_json(raw)

    def embedder(self, model: str = "text-embedding-3-small"):
        """Return a batched embedder that reuses this backend's client.

        Used for *semantic* grow-and-refine de-duplication. Sharing the client
        means it inherits the same api_key, base_url, retries, and timeout.
        """
        client = self._client

        def _embed(texts):
            resp = client.embeddings.create(model=model, input=list(texts))
            return [d.embedding for d in resp.data]

        return _embed


class SimulatedLLM:
    """Deterministic, offline backend backed by a teaching environment.

    The simulated model behaves like a base LLM that *innately knows* a fraction
    of the domain rules. For any other question it is correct only if the
    relevant rule is present in the context (playbook) it is given. The Reflector
    and Curator simulations extract and itemize those rules from feedback, so ACE
    measurably improves over time. This is a simulation of the ACE *mechanism*,
    not a substitute for real benchmarks.
    """

    def __init__(self, environment, seed: int = 0) -> None:
        self.env = environment
        self.seed = seed
        self.num_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0

    # The simulated roles are driven directly by ace.roles via these helpers,
    # but we also expose the LLM protocol so the type checks and any custom
    # prompting paths still work.
    def complete(self, system: str, user: str, **kwargs) -> str:  # pragma: no cover
        self.num_calls += 1
        return ""

    def complete_json(self, system: str, user: str, **kwargs) -> dict:  # pragma: no cover
        self.num_calls += 1
        return {}
