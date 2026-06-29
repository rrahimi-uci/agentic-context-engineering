"""Tests for every cookbook recipe.

The core recipes (01–07) run fully offline and are exercised end-to-end: we call
their ``run()`` and assert on the returned results. The agent recipes (08–10)
need an API key, so here we assert they import cleanly and that ``main()`` exits
gracefully without one — their underlying behavior is covered by
``test_openai_agents_integration.py``.

Recipes live in ``cookbook/`` (not an importable package — the filenames start
with digits), so we load each module by path with importlib.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

# --- optional Allure annotations (no-op if allure-pytest isn't installed) --- #
try:  # pragma: no cover - trivial shim
    import allure

    _feature = allure.feature
    _story = allure.story
except Exception:  # pragma: no cover

    def _noop_decorator_factory(*_args, **_kwargs):
        def _decorate(fn):
            return fn

        return _decorate

    _feature = _story = _noop_decorator_factory


COOKBOOK = pathlib.Path(__file__).resolve().parent.parent / "cookbook"


def _load(filename: str):
    """Import a cookbook recipe module by path."""
    path = COOKBOOK / filename
    mod_name = "cookbook_" + filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader, f"could not load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Sanity: the cookbook directory is well-formed
# --------------------------------------------------------------------------- #
def test_cookbook_directory_exists():
    assert COOKBOOK.is_dir()
    assert (COOKBOOK / "README.md").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "01_first_playbook.py",
        "02_online_adaptation.py",
        "03_your_own_task.py",
        "04_label_free_feedback.py",
        "05_save_and_resume.py",
        "06_grow_and_refine.py",
        "07_inspect_and_report.py",
        "08_agent_quickstart.py",
        "09_agent_auto_learn_from_tool_errors.py",
        "10_agent_streaming_and_sessions.py",
    ],
)
def test_recipe_imports_and_exposes_main(filename):
    module = _load(filename)
    assert hasattr(module, "main"), f"{filename} must define main()"


# --------------------------------------------------------------------------- #
# Core recipes (01–07): exercised end-to-end
# --------------------------------------------------------------------------- #
@_feature("Cookbook — core")
@_story("01 first playbook")
def test_01_first_playbook_beats_baseline():
    r = _load("01_first_playbook.py").run()
    assert r["playbook_bullets"] >= 1
    assert r["ace_accuracy"] >= r["base_accuracy"]
    assert r["ace_accuracy"] > 50.0


@_feature("Cookbook — core")
@_story("02 online adaptation")
def test_02_online_adaptation_improves_over_time():
    r = _load("02_online_adaptation.py").run()
    assert r["playbook_bullets"] >= 1
    assert r["late_accuracy"] >= r["early_accuracy"]
    assert r["growth_curve"][-1] >= r["growth_curve"][0]


@_feature("Cookbook — core")
@_story("03 your own task")
def test_03_your_own_task_learns():
    r = _load("03_your_own_task.py").run()
    assert r["task_name"] == "my-domain"
    assert r["num_samples"] > 0
    assert r["playbook_bullets"] >= 1


@_feature("Cookbook — core")
@_story("04 label-free feedback")
def test_04_label_free_feedback_learns():
    r = _load("04_label_free_feedback.py").run()
    assert r["playbook_bullets"] >= 1
    assert r["graded_steps"] > 0
    assert r["accuracy"] > 0.0


@_feature("Cookbook — core")
@_story("05 save and resume")
def test_05_save_and_resume_roundtrips(tmp_path):
    path = str(tmp_path / "pb.json")
    r = _load("05_save_and_resume.py").run(path=path)
    assert r["saved_bullets"] == r["resumed_bullets"]
    assert r["resumed_bullets"] >= 1
    assert r["resumed_accuracy"] > 0.0


@_feature("Cookbook — core")
@_story("06 grow and refine")
def test_06_grow_and_refine_dedupes_and_prunes():
    r = _load("06_grow_and_refine.py").run()
    assert r["after"] < r["before"]
    assert len(r["deduped"]) >= 1
    assert len(r["pruned"]) >= 1
    # The harmful bullet must be gone; a useful, distinct one must remain.
    assert any("arrival estimate" in s for s in r["survivors"])
    assert not any("Skip identity checks" in s for s in r["survivors"])


@_feature("Cookbook — core")
@_story("07 inspect and report")
def test_07_inspect_and_report_writes_html(tmp_path):
    path = str(tmp_path / "report.html")
    r = _load("07_inspect_and_report.py").run(report_path=path)
    assert r["num_bullets"] >= 1
    assert r["approx_tokens"] > 0
    assert r["report_is_html"]
    assert r["report_bytes"] > 500
    assert pathlib.Path(path).exists()


# --------------------------------------------------------------------------- #
# Agent recipes (08–10): import-clean and graceful without a key
# --------------------------------------------------------------------------- #
@_feature("Cookbook — agents")
@pytest.mark.parametrize(
    "filename",
    [
        "08_agent_quickstart.py",
        "09_agent_auto_learn_from_tool_errors.py",
        "10_agent_streaming_and_sessions.py",
    ],
)
def test_agent_recipes_exit_cleanly_without_key(filename, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = _load(filename)
    assert module.main() == 0
