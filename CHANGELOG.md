# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Quality gates** — `ruff` (lint + format), `mypy` (backing `py.typed`), and
  `pytest-cov` (≥90% floor, currently ~95%), enforced by a new CI `quality` job
  and the coverage-gated integration job.
- **`OpenAILLM` hardening** — native JSON via `response_format={"type":"json_object"}`
  (with transparent fallback for providers that reject it), built-in
  `max_retries`/backoff and a request `timeout`.
- **Non-blocking async** — `ACEAgent.arun_and_learn` / `arun_streamed_and_learn`
  offload the synchronous Reflector/Curator step to a worker thread
  (`asyncio.to_thread`), keeping the caller's event loop responsive.
- **Token/cost observability** — `StepRecord` and `RunResult.summary()` now report
  `llm_calls`, prompt/completion tokens, and `cached_prompt_tokens` (OpenAI's
  automatic prefix cache of the static system + playbook prefix).
- **Parallel `evaluate(max_workers=…)`** — concurrent, order-preserving,
  result-identical inference pass.
- **Auto-wired semantic embedder** — when no embedder is passed and a role
  backend is an `OpenAILLM`, grow-and-refine de-duplication uses OpenAI
  embeddings (config `auto_embedder`, lexical fallback on any error).
- `CHANGELOG.md`.

### Changed
- All GitHub Actions pinned to Node 24 majors (the Sept-2025 Node 20 deprecation).
- `__version__` is now sourced from package metadata (single source of truth in
  `pyproject.toml`).

## [0.2.0] - 2026-06-29

### Added
- **First-class OpenAI Agents SDK integration**: `wrap_agent()` one-call entry
  point with playbook injection, persistence, auto-learn from tool errors
  (`RunHooks`), typed-run-item trajectory capture, an `ace.learn` tracing span,
  and streaming/async support. Top-level lazy re-exports (`from ace import
  wrap_agent, ACEAgent`).
- **Cookbook** — 10 guided, tested recipes (7 offline + 3 agent) and a docs page.
- Allure test reporting published as a CI artifact.

### Changed
- `openai-agents` pinned to a realistic floor (`>=0.1,<1.0`).

## [0.1.0] - 2026-06-27

### Added
- Initial release: the ACE engine (Generator → Reflector → Curator, incremental
  delta merge, grow-and-refine), `OpenAILLM` + deterministic `SimulatedLLM`
  backends, the `ace` CLI, a live terminal dashboard, and a self-contained HTML
  report. A faithful, dependency-light implementation of the ICLR 2026 paper.

[Unreleased]: https://github.com/rrahimi-uci/agentic-context-engineering/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rrahimi-uci/agentic-context-engineering/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rrahimi-uci/agentic-context-engineering/releases/tag/v0.1.0
