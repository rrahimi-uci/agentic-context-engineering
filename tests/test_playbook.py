from ace.playbook import Bullet, Playbook


def test_add_and_get():
    pb = Playbook()
    b = pb.add(Bullet(content="hello", section="strategies"))
    assert len(pb) == 1
    assert pb.get(b.id) is b
    assert b.id in pb


def test_update_and_remove():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    assert pb.update(b.id, "y") is True
    assert pb.get(b.id).content == "y"
    assert pb.update("nope", "z") is False
    assert pb.remove(b.id) is True
    assert pb.remove(b.id) is False
    assert len(pb) == 0


def test_counters_and_score():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    pb.mark_helpful(b.id, 3)
    pb.mark_harmful(b.id, 1)
    assert pb.get(b.id).score == 2


def test_render_groups_by_section():
    pb = Playbook()
    pb.add(Bullet(content="strat", section="strategies"))
    pb.add(Bullet(content="concept", section="domain_concepts"))
    rendered = pb.render()
    assert "Strategies" in rendered
    assert "Domain Concepts" in rendered
    assert "strat" in rendered and "concept" in rendered


def test_empty_render():
    assert "empty" in Playbook().render().lower()


def test_unknown_section_appended():
    pb = Playbook(sections=["strategies"])
    pb.add(Bullet(content="x", section="brand_new"))
    assert "brand_new" in pb.sections


def test_serialization_roundtrip():
    pb = Playbook()
    pb.add(Bullet(content="a", section="strategies", helpful_count=2))
    pb.add(Bullet(content="b", section="formatting"))
    d = pb.to_dict()
    pb2 = Playbook.from_dict(d)
    assert len(pb2) == 2
    assert pb2.bullets[0].helpful_count == 2
    assert pb2.bullets[1].section == "formatting"


def test_save_load(tmp_path):
    pb = Playbook()
    pb.add(Bullet(content="persist me"))
    p = tmp_path / "pb.json"
    pb.save(str(p))
    pb2 = Playbook.load(str(p))
    assert pb2.bullets[0].content == "persist me"


def test_stats_and_tokens():
    pb = Playbook()
    pb.add(Bullet(content="a" * 100))
    stats = pb.stats()
    assert stats["num_bullets"] == 1
    assert stats["approx_tokens"] >= 1


def test_clone_is_independent():
    pb = Playbook()
    b = pb.add(Bullet(content="x"))
    clone = pb.clone()
    clone.update(b.id, "changed")
    assert pb.get(b.id).content == "x"
