# Contributing to ACE

Thanks for your interest! This project is a faithful, open-source implementation
of Agentic Context Engineering and welcomes improvements.

## Getting started

```bash
git clone https://github.com/your-org/ace-framework && cd ace-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ground rules

- **Keep the core dependency-light.** The library proper depends only on
  `numpy` and `rich`. Anything needing `openai` / `openai-agents` lives behind
  an optional extra and an import guard.
- **Everything must run offline.** Tests and demos use `SimulatedLLM` +
  `TeachingEnvironment` and must not require a network or API key.
- **The merge stays deterministic.** `apply_delta` and `grow_and_refine` must
  never call an LLM — that property is what prevents context collapse.
- **Add tests.** New behavior needs coverage; the suite runs in well under a second.

## Good first issues

- Additional agent-framework integrations (LangGraph, CrewAI, AutoGen).
- More `Task` adapters for real benchmarks (AppWorld, FiNER, BIRD-SQL).
- Alternative embedders for semantic de-duplication.
- A `matplotlib`/`plotly` exporter alongside the built-in SVG report.

## Pull requests

1. Branch from `main`.
2. `pytest` must pass.
3. Describe *what* changed and *why* in the PR body.
