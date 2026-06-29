"""Recipe 08 — Give an OpenAI Agents SDK agent a self-improving memory.

What you'll learn
-----------------
* The crystal-clear one-call entry point: ``wrap_agent`` builds the ACE engine,
  injects the playbook into the agent's instructions on every run, and persists
  what it learns.
* How to feed back a natural-language ``signal`` after a run so the agent writes
  itself a durable rule — no ground-truth labels required.

Requirements
------------
A real OpenAI API key and the extras::

    pip install "ace-playbook[all]"
    export OPENAI_API_KEY=sk-...
    python cookbook/08_agent_quickstart.py

Without a key the recipe prints a note and exits cleanly (so it stays importable
and CI-safe). The underlying API is covered by the integration test suite.
"""

from __future__ import annotations

import os


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and `pip install ace-playbook[all]`) to run this recipe.")
        return 0

    from agents import Agent, function_tool

    from ace import wrap_agent

    @function_tool
    def lookup_order(order_id: str) -> str:
        """Look up an order's status by id."""
        return f"Order {order_id}: shipped, arriving in 2 days."

    base = Agent(
        name="SupportAgent",
        instructions="You are a concise customer-support agent. Use tools when helpful.",
        tools=[lookup_order],
        model="gpt-4o-mini",
    )

    # One call: builds ACE(OpenAILLM(...)), loads support_memory.json if present,
    # and remembers the path as the default save target.
    agent = wrap_agent(base, model="gpt-4o-mini", playbook="support_memory.json")

    lessons = [
        ("Where is order #A17?", "Always call lookup_order before answering status questions."),
        (
            "Customer asks to cancel #C99 — what do you do?",
            "Cancellation requires confirming identity first; never cancel without verification.",
        ),
    ]
    for question, signal in lessons:
        out = agent.run_and_learn(question, signal=signal)
        print(f"\nQ: {question}\nA: {out.output}")
        print(f"   playbook now {out.record.playbook_size} bullets")

    print("\n=== Learned playbook ===")
    print(agent.playbook.render())
    print(f"\nSaved to {agent.save()} — re-run to keep getting smarter.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
