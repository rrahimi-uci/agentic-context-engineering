"""Command-line interface: ``ace``.

Commands
--------
* ``ace demo``      — run the headline comparison (Base vs ACE vs Monolithic),
                      print a table, and optionally write an HTML report.
* ``ace run``       — adapt offline+online on the teaching task with a live view.
* ``ace report``    — (re)generate an HTML report from a saved run JSON.
* ``ace playbook``  — pretty-print a saved playbook.
* ``ace version``   — print the version.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import __version__
from .baselines import MonolithicRewriteAgent, StaticAgent
from .config import ACEConfig
from .engine import ACE
from .llm import SimulatedLLM
from .playbook import Playbook
from .tasks import TeachingEnvironment, build_teaching_task
from .visualize import LiveRunVisualizer, save_html_report


def _print_table(rows, headers):
    from rich.console import Console
    from rich.table import Table

    t = Table(show_header=True, header_style="bold cyan")
    for h in headers:
        t.add_column(h)
    for r in rows:
        t.add_row(*[str(x) for x in r])
    Console().print(t)


def cmd_demo(args: argparse.Namespace) -> int:
    from rich.console import Console

    console = Console()
    console.rule("[bold]ACE — headline comparison (offline teaching environment)[/bold]")
    env = TeachingEnvironment(known_fraction=args.known_fraction, seed=args.seed)
    task = build_teaching_task(repeats=args.repeats, seed=args.seed)
    train, test = task.split(train_frac=0.5, seed=args.seed)

    base = StaticAgent(SimulatedLLM(env)).run(test)

    ace = ACE(SimulatedLLM(env), ACEConfig(epochs=args.epochs))
    ace.adapt_offline(train)
    ace_eval = ace.evaluate(test)

    mono = MonolithicRewriteAgent(SimulatedLLM(env)).run(task)
    ace_online = ACE(SimulatedLLM(env)).adapt_online(task)

    rows = [
        ("Base LLM (no context)", f"{base.accuracy:.1f}%", "0", "—"),
        (
            "ACE (offline → eval)",
            f"{ace_eval.accuracy:.1f}%",
            str(len(ace.playbook)),
            f"+{ace_eval.accuracy - base.accuracy:.1f} pts",
        ),
        (
            "Monolithic rewrite (online)",
            f"{mono.accuracy:.1f}%",
            str(mono.history[-1].playbook_size),
            f"{sum(1 for r in mono.history if r.refine.get('collapsed'))} collapses",
        ),
        (
            "ACE (online)",
            f"{ace_online.accuracy:.1f}%",
            str(len(ace_online.playbook.bullets) if ace_online.playbook else 0),
            "no collapse",
        ),
    ]
    _print_table(rows, ["Method", "Accuracy", "Playbook", "Note"])

    if args.html:
        path = save_html_report(
            {"ACE (online)": ace_online, "Monolithic rewrite": mono},
            out_path=args.html,
            subtitle="Base vs ACE vs Monolithic rewrite (context collapse)",
        )
        console.print(f"\n[green]HTML report written to[/green] {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    env = TeachingEnvironment(known_fraction=args.known_fraction, seed=args.seed)
    task = build_teaching_task(repeats=args.repeats, seed=args.seed)
    ace = ACE(SimulatedLLM(env), ACEConfig(epochs=args.epochs))
    title = "ACE online adaptation"
    with LiveRunVisualizer(title=title, total=len(task.samples)) as viz:
        result = ace.adapt_online(task, callback=viz)
    from rich.console import Console

    Console().print(
        f"\nFinal accuracy: [green]{result.accuracy:.1f}%[/green] · "
        f"playbook: {len(ace.playbook)} bullets"
    )
    if args.save_playbook:
        ace.playbook.save(args.save_playbook)
        Console().print(f"Playbook saved to {args.save_playbook}")
    if args.html:
        save_html_report({"ACE (online)": result}, out_path=args.html)
        Console().print(f"HTML report written to {args.html}")
    return 0


def cmd_playbook(args: argparse.Namespace) -> int:
    pb = Playbook.load(args.path)
    print(pb.render())
    print("\n" + json.dumps(pb.stats(), indent=2))
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"ace {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ace", description="Agentic Context Engineering CLI")
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--repeats", type=int, default=3, help="rule-bank cycles")
        sp.add_argument("--epochs", type=int, default=5, help="offline epochs")
        sp.add_argument("--known-fraction", type=float, default=0.35, dest="known_fraction")
        sp.add_argument("--seed", type=int, default=1)

    d = sub.add_parser("demo", help="headline comparison with numbers")
    common(d)
    d.add_argument("--html", default=None, help="write an HTML report to this path")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="live online adaptation run")
    common(r)
    r.add_argument("--html", default=None)
    r.add_argument("--save-playbook", default=None, dest="save_playbook")
    r.set_defaults(func=cmd_run)

    pb = sub.add_parser("playbook", help="print a saved playbook")
    pb.add_argument("path")
    pb.set_defaults(func=cmd_playbook)

    v = sub.add_parser("version", help="print version")
    v.set_defaults(func=cmd_version)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
