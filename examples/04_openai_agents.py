"""Use ACE as a self-improving memory for an OpenAI Agents SDK agent.

Requires a real OpenAI API key (set OPENAI_API_KEY) and the extras:
    pip install "ace-playbook[all]"

The agent runs normally; after each task we feed natural execution feedback
(no labels required) to ACE, which grows a playbook that is injected into the
agent's instructions on subsequent runs. The playbook is persisted to disk, so
re-running this script starts from what the agent already learned.

Run:  OPENAI_API_KEY=sk-... python examples/04_openai_agents.py
"""

import os
import sys


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this live demo.")
        return 0

    from agents import Agent, function_tool

    from ace import wrap_agent  # one import — the crystal-clear entry point

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

    # wrap_agent builds the ACE engine (same model family for Reflector/Curator),
    # loads support_memory.json if it exists, and persists what it learns.
    agent = wrap_agent(base, model="gpt-4o-mini", playbook="support_memory.json")

    tasks = [
        ("Where is order #A17?", "Always call lookup_order before answering status questions."),
        ("Is order #B22 delayed?", "State the arrival estimate explicitly and reassure the customer."),
        ("Customer asks to cancel #C99 — what do you do?",
         "Cancellation requires confirming identity first; never cancel without verification."),
    ]

    for q, signal in tasks:
        out = agent.run_and_learn(q, signal=signal)
        print(f"\nQ: {q}\nA: {out.output}\n   (playbook now {out.record.playbook_size} bullets)")

    print("\n=== Learned playbook ===")
    print(agent.playbook.render())
    path = agent.save()  # persists to support_memory.json
    print(f"\nPlaybook saved to {path} — re-run to continue learning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
