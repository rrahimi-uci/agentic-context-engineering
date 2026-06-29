"""Tests for the ``ace`` command-line interface (all offline, no API key)."""

from ace import __version__
from ace.cli import build_parser, main
from ace.playbook import Bullet, Playbook


def test_version_command(capsys):
    rc = main(["version"])
    assert rc == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_demo_runs_and_prints_table(capsys):
    rc = main(["demo", "--repeats", "1", "--epochs", "1", "--seed", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Base LLM" in out
    assert "ACE" in out


def test_demo_writes_html(tmp_path, capsys):
    out_path = tmp_path / "report.html"
    rc = main(["demo", "--repeats", "1", "--epochs", "1", "--html", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert "<!DOCTYPE html>" in out_path.read_text(encoding="utf-8")


def test_run_command_offline(tmp_path, capsys):
    pb_path = tmp_path / "pb.json"
    rc = main(["run", "--repeats", "1", "--epochs", "1", "--save-playbook", str(pb_path)])
    assert rc == 0
    assert pb_path.exists()


def test_playbook_command(tmp_path, capsys):
    pb = Playbook()
    pb.add(Bullet(content="a learned rule", section="strategies"))
    path = tmp_path / "pb.json"
    pb.save(str(path))

    rc = main(["playbook", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "a learned rule" in out
    # The stats JSON blob is appended.
    assert "num_bullets" in out


def test_build_parser_exposes_subcommands():
    parser = build_parser()
    # Parsing a known subcommand should set a dispatch function.
    args = parser.parse_args(["version"])
    assert callable(args.func)
