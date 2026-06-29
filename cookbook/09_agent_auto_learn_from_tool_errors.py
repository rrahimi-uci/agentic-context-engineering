"""Recipe 09 — Auto-learn from tool failures (no explicit feedback).

What you'll learn
-----------------
* With ``capture=True`` (the default), ``wrap_agent`` attaches a ``RunHooks``
  listener that records the real tool trajectory. If a tool *errors* and you give
  no explicit feedback, that error automatically becomes the execution signal —
  so the agent learns to avoid the failure next time.
* Where to find the captured signal and event log: ``out.auto_signal`` and
  ``out.events``.

Requirements
------------
A real OpenAI API key and the extras::

    pip install "ace-playbook[all]"
    export OPENAI_API_KEY=sk-...
    python cookbook/09_agent_auto_learn_from_tool_errors.py

Without a key the recipe prints a note and exits cleanly (CI-safe). The hooks /
auto-signal behavior itself is covered by the integration test suite.
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
    def charge_card(amount: str) -> str:
        """Charge the customer's card (simulated to fail without a confirmed amount)."""
        if not amount.strip().isdigit():
            return "Error: payment gateway returned 400 — amount must be a confirmed integer."
        return f"Charged ${amount}.00 successfully (200 OK)."

    base = Agent(
        name="BillingAgent",
        instructions="You are a billing agent. Use charge_card to process payments.",
        tools=[charge_card],
        model="gpt-4o-mini",
    )
    agent = wrap_agent(base, model="gpt-4o-mini")  # capture=True by default

    # Note: no `signal=` and no label. If charge_card errors, the captured error
    # becomes the signal and the Reflector turns it into a lesson automatically.
    out = agent.run_and_learn("Charge the customer for the order.")
    print("Answer        :", out.output)
    print("Auto signal   :", out.auto_signal or "(no tool error captured)")
    print("Captured events:")
    for ev in out.events:
        print("   ", ev)
    print(f"\nPlaybook now {len(agent.playbook)} bullets:")
    print(agent.playbook.render())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
