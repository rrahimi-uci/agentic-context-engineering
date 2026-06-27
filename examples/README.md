# Examples

Runnable demos. The first three need **no API key** (deterministic offline `SimulatedLLM`).

| File | What it shows | API key |
|---|---|---|
| `01_quickstart.py` | ACE in ~10 lines: base vs offline-adapted accuracy | no |
| `02_context_collapse.py` | Reproduces *context collapse*; ACE vs monolithic rewrite; writes `ace_report.html` | no |
| `03_offline_vs_online.py` | Offline warmup + online adaptation; delta-vs-rewrite token cost | no |
| `04_openai_agents.py` | ACE as self-improving memory for an OpenAI Agents SDK agent | yes |

```bash
python examples/01_quickstart.py
```
