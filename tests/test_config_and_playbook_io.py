"""Tests for ACEConfig defaults and Playbook (de)serialization."""

from ace.config import ACEConfig
from ace.playbook import DEFAULT_SECTIONS, Bullet, Playbook


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_defaults_are_paper_aligned():
    cfg = ACEConfig()
    assert cfg.reflector_max_rounds == 5
    assert cfg.epochs == 5
    assert cfg.curator_use_llm is True
    assert cfg.use_labels is True
    assert cfg.sections == DEFAULT_SECTIONS
    # The section list is a *copy*, not the shared module-level default.
    cfg.sections.append("extra")
    assert "extra" not in DEFAULT_SECTIONS


def test_config_is_overridable():
    cfg = ACEConfig(epochs=2, curator_use_llm=False, lazy_refine_token_budget=100)
    assert cfg.epochs == 2
    assert cfg.curator_use_llm is False
    assert cfg.lazy_refine_token_budget == 100


# --------------------------------------------------------------------------- #
# Playbook IO
# --------------------------------------------------------------------------- #
def test_playbook_save_load_roundtrip(tmp_path):
    pb = Playbook()
    b = pb.add(Bullet(content="rule one", section="strategies", tags=["t"]))
    pb.mark_helpful(b.id, 3)
    pb.mark_harmful(b.id, 1)
    path = tmp_path / "pb.json"
    pb.save(str(path))

    loaded = Playbook.load(str(path))
    assert len(loaded) == 1
    rb = loaded.get(b.id)
    assert rb is not None
    assert rb.content == "rule one"
    assert rb.helpful_count == 3 and rb.harmful_count == 1
    assert rb.score == 2


def test_playbook_clone_is_independent():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    clone = pb.clone()
    clone.remove(b.id)
    assert len(pb) == 1  # original untouched
    assert len(clone) == 0


def test_playbook_render_empty():
    assert "empty" in Playbook().render().lower()


def test_playbook_add_appends_unknown_section():
    # add() is permissive: an unknown section is appended to the section list
    # and rendered under its own (title-cased) header.
    pb = Playbook(sections=["strategies"])
    pb.add(Bullet(content="weird", section="not_a_known_section"))
    rendered = pb.render()
    assert "Not A Known Section" in rendered
    assert "weird" in rendered
    assert "not_a_known_section" in pb.sections


def test_playbook_renders_orphan_section_under_other():
    # If a bullet's section is somehow not in the section list, render() puts it
    # under an "Other" heading (defensive path).
    pb = Playbook(sections=["strategies"])
    b = Bullet(content="orphan", section="strategies")
    pb._bullets[b.id] = b  # bypass add(): keep section list unchanged
    pb.sections = ["formatting"]  # now b.section is not listed
    rendered = pb.render()
    assert "Other" in rendered
    assert "orphan" in rendered


def test_playbook_stats_and_tokens():
    pb = Playbook()
    pb.add(Bullet(content="a" * 40, section="strategies"))
    stats = pb.stats()
    assert stats["num_bullets"] == 1
    assert stats["approx_tokens"] >= 1
    assert pb.approx_tokens() >= 1


def test_bullet_from_dict_ignores_unknown_keys():
    b = Bullet.from_dict({"content": "c", "section": "strategies", "bogus": 1})
    assert b.content == "c"
    assert not hasattr(b, "bogus")
