# Examples

Runnable demos. All but the last need **no API key** (deterministic offline `SimulatedLLM`).

| File | What it shows | API key |
|---|---|---|
| `01_quickstart.py` | ACE in ~10 lines: base vs offline-adapted accuracy | no |
| `02_context_collapse.py` | Reproduces *context collapse*; ACE vs monolithic rewrite; writes `ace_report.html` | no |
| `03_offline_vs_online.py` | Offline warmup + online adaptation; delta-vs-rewrite token cost | no |
| `04_openai_agents.py` | ACE as self-improving memory for an OpenAI Agents SDK agent (`wrap_agent`) | yes |
| `05_custom_task.py` | Bring your own `Task` + `feedback_fn` (the general-purpose path) | no |

```bash
python examples/01_quickstart.py
```

> Looking for a guided, concept-by-concept tour? See the **[Cookbook](../cookbook/README.md)** —
> ten short recipes (with tests) covering offline/online adaptation, custom tasks,
> label-free feedback, persistence, grow-and-refine, and the OpenAI Agents SDK.
