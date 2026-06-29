"""Recipe 10 — Streaming, async, and multi-turn sessions.

What you'll learn
-----------------
* Stream a run and learn from it at completion with
  ``await agent.arun_streamed_and_learn(query, on_event=...)`` — ideal inside an
  event loop (FastAPI, notebooks).
* That **ACE memory and SDK sessions are orthogonal and compose**:
    - ACE playbook = *cross-task learned strategy* (what the agent figured out).
    - SDK ``session=`` = *within-conversation history* (what was said this chat).
  Pass a ``session`` straight through any run method.

Requirements
------------
A real OpenAI API key and the extras::

    pip install "ace-playbook[all]"
    export OPENAI_API_KEY=sk-...
    python cookbook/10_agent_streaming_and_sessions.py

Without a key the recipe prints a note and exits cleanly (CI-safe). Streaming and
session passthrough are covered by the integration test suite.
"""

from __future__ import annotations

import asyncio
import os


async def _demo() -> None:
    from agents import Agent, SQLiteSession

    from ace import wrap_agent

    agent = wrap_agent(
        Agent(name="Assistant", instructions="You are a helpful, concise assistant."),
        model="gpt-4o-mini",
    )

    # A session gives the conversation turn-to-turn memory; ACE gives it
    # cross-conversation, self-improving strategy. They stack cleanly.
    session = SQLiteSession("cookbook-demo")

    printed: list[str] = []

    def on_event(event) -> None:
        # Called for every streamed event; here we just count them.
        printed.append(type(event).__name__)

    out = await agent.arun_streamed_and_learn(
        "Summarize the benefits of itemized agent memory in one sentence.",
        signal="Be concise and concrete.",
        session=session,
        on_event=on_event,
    )
    print("Streamed answer :", out.output)
    print("Stream events   :", len(printed))
    print("Playbook bullets:", len(agent.playbook))


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and `pip install ace-playbook[all]`) to run this recipe.")
        return 0
    asyncio.run(_demo())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
