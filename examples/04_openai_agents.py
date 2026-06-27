"""Use ACE as a self-improving memory for an OpenAI Agents SDK agent.

Requires a real OpenAI API key (set OPENAI_API_KEY) and the extras:
    pip install "agentic-context-engineering[all]"

The agent runs normally; after each task we feed natural execution feedback
(no labels required) to ACE, which grows a playbook that is injected into the
agent's instructions on subsequent runs.

Run:  OPENAI_API_KEY=sk-... python examples/04_openai_agents.py
"""

import os
import sys


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this live demo.")
        return 0

    from agents import Agent, function_tool

    from ace import ACE, ACEConfig, OpenAILLM
    from ace.integrations.openai_agents import ACEAgent

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

    # ACE uses the same model family for Reflector/Curator.
    llm = OpenAILLM(model="gpt-4o-mini")
    agent = ACEAgent(base, ace=ACE(llm, ACEConfig(reflector_max_rounds=1)))

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
    print(agent.ace.playbook.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
