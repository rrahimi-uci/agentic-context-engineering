<div align="center">

# 📓 ACE Cookbook

### Short, runnable recipes that teach the library one concept at a time.

Every recipe is a self-contained script with a `run()` you can import and a
`main()` you can execute. Recipes **01–07 need no API key** (they use the
deterministic `SimulatedLLM`), so they run anywhere — including CI — and are
covered by [`tests/test_cookbook.py`](../tests/test_cookbook.py).

[Core recipes](#-core-recipes-no-api-key) · [Agent recipes](#-openai-agents-sdk-recipes-api-key) · [Learning path](#-suggested-learning-path) · [Run them](#-running-the-recipes)

</div>

---

## 🧪 Core recipes (no API key)

| # | Recipe | What it teaches |
|---|--------|-----------------|
| 01 | [`01_first_playbook.py`](01_first_playbook.py) | The 3-line workflow — construct → `adapt_offline` → `evaluate` — and the lift over a base LLM. |
| 02 | [`02_online_adaptation.py`](02_online_adaptation.py) | Test-time learning: predict-then-learn per sample; proving improvement with windowed accuracy. |
| 03 | [`03_your_own_task.py`](03_your_own_task.py) | Defining a `Task` from your own `Sample`s and scorer — ACE is domain-agnostic. |
| 04 | [`04_label_free_feedback.py`](04_label_free_feedback.py) | Learning with **no gold labels** via a `feedback_fn` that returns execution signals. |
| 05 | [`05_save_and_resume.py`](05_save_and_resume.py) | Persisting a playbook to JSON and warm-starting a fresh engine from it. |
| 06 | [`06_grow_and_refine.py`](06_grow_and_refine.py) | De-duplicating and pruning a playbook with `grow_and_refine` — keeping it compact. |
| 07 | [`07_inspect_and_report.py`](07_inspect_and_report.py) | Introspecting bullets/stats and rendering a shareable **HTML report**. |

## 🔌 OpenAI Agents SDK recipes (API key)

Install the extras and set a key: `pip install "ace-playbook[all]" && export OPENAI_API_KEY=sk-...`
Each recipe exits cleanly with a friendly note if the key is missing, so they stay
import-safe in CI.

| # | Recipe | What it teaches |
|---|--------|-----------------|
| 08 | [`08_agent_quickstart.py`](08_agent_quickstart.py) | `wrap_agent` in one call: playbook injection, `run_and_learn`, and `save()`. |
| 09 | [`09_agent_auto_learn_from_tool_errors.py`](09_agent_auto_learn_from_tool_errors.py) | Auto-learning from tool failures via capture hooks — `out.auto_signal` / `out.events`. |
| 10 | [`10_agent_streaming_and_sessions.py`](10_agent_streaming_and_sessions.py) | Streaming + async (`arun_streamed_and_learn`) and composing with an SDK `session`. |

---

## 🗺️ Suggested learning path

```text
01 first_playbook ─► 02 online_adaptation ─► 03 your_own_task ─► 04 label_free_feedback
        │                                                              │
        ▼                                                              ▼
05 save_and_resume ─► 06 grow_and_refine ─► 07 inspect_and_report      │
                                                                       ▼
                              08 agent_quickstart ─► 09 auto_learn ─► 10 streaming/sessions
```

1. **Start with 01–02** to internalize the offline vs. online regimes.
2. **03–04** show how to point ACE at *your* data and *your* feedback.
3. **05–07** are the operational concerns: persistence, compaction, observability.
4. **08–10** wire all of the above into a real OpenAI Agents SDK agent.

---

## ▶️ Running the recipes

```bash
# Core recipes — no key, fully deterministic
python cookbook/01_first_playbook.py
python cookbook/06_grow_and_refine.py

# Agent recipes — need the extras + a key
pip install "ace-playbook[all]"
export OPENAI_API_KEY=sk-...
python cookbook/08_agent_quickstart.py
```

Every recipe is verified by the test suite:

```bash
pip install -e ".[dev]"
pytest tests/test_cookbook.py -q
```

---

## 🧩 Recipe anatomy

Each recipe follows the same shape so it's easy to read **and** to test:

```python
def run() -> dict:
    """The recipe's logic. Returns the results so tests can assert on them."""
    ...

def main() -> int:
    """Pretty-prints run() and returns an exit code. Used when run as a script."""
    ...
```

Tests import `run()` and assert on its return value; running the file as a script
calls `main()`.

---

<div align="center">
<sub>New to ACE? Start at the <a href="../README.md">project README</a> and the
<a href="../ARCHITECTURE.md">architecture guide</a>.</sub>
</div>
