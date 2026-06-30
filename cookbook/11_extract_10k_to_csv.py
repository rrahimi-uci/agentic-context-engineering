"""Recipe 11 — A 10-K extraction agent that ports to CSV, optimized by ACE.

What you'll learn
-----------------
* How to wrap a *real-world* job — reading pages of a company's annual report
  (SEC Form **10-K**) and extracting structured financials into a **CSV** — as an
  ACE :class:`~ace.tasks.Task`.
* How ACE turns the agent's mistakes into reusable **extraction rules** (one
  bullet per field: *where* the number lives, *which* look-alike line items to
  avoid). The agent learns the rules from a couple of filings and applies them to
  a **brand-new company's 10-K it has never seen**.
* That the same loop that fixes a multiple-choice benchmark fixes a messy
  document-extraction pipeline — the engine is domain-agnostic.

The story
---------
A 10-K page is full of look-alike numbers: prior-year columns, segment
subtotals, "Total *current* assets" vs. "Total assets", Basic vs. Diluted EPS,
the 21% statutory rate vs. the real effective rate. A naive extractor grabs the
wrong one. We frame each field as "pick the correct value from the candidates on
the page" so the recipe stays **deterministic and needs no API key** — the
simulated agent is wrong until ACE teaches it the right line item, then it nails
that field across all companies.

    python cookbook/11_extract_10k_to_csv.py        # writes a CSV of extractions

Going real
----------
Swap the deterministic backend for a real model and this becomes a genuine
report-reading agent — the surrounding ACE code is identical::

    from ace import ACE, OpenAILLM
    ace = ACE(OpenAILLM(model="gpt-4o-mini"))       # reads real 10-K text
    ace.adapt_offline(train, feedback_fn=my_checker)

or plug the learned playbook into an OpenAI Agents SDK agent with
``wrap_agent(...)`` (see recipes 08-10) so a tool-using agent improves its
extraction prompt from execution feedback.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from typing import Dict, List, Tuple

from ace import ACE, ACEConfig, Sample, SimulatedLLM, Task, TeachingEnvironment
from ace.baselines import StaticAgent

# --------------------------------------------------------------------------- #
# The fields we want in the CSV, each with the extraction *rule* ACE should learn
# and the trap that makes a naive agent get it wrong.
# --------------------------------------------------------------------------- #
FIELDS: List[Dict[str, str]] = [
    {
        "concept": "total_revenue",
        "label": "Total revenue (USD millions)",
        "rule": "Total revenue is the 'Total net sales' line on the Consolidated "
        "Statements of Operations for the most recent fiscal year — not the "
        "prior-year column and not a segment subtotal.",
    },
    {
        "concept": "net_income",
        "label": "Net income (USD millions)",
        "rule": "Net income is the bottom-line 'Net income' after taxes — not "
        "'Operating income' and not 'Income before income taxes'.",
    },
    {
        "concept": "total_assets",
        "label": "Total assets (USD millions)",
        "rule": "Total assets is the final total of the assets section on the "
        "Consolidated Balance Sheets — not 'Total current assets'.",
    },
    {
        "concept": "diluted_eps",
        "label": "Diluted EPS (USD)",
        "rule": "Diluted EPS is reported under 'Earnings per share — Diluted' at "
        "the bottom of the income statement, not Basic EPS.",
    },
    {
        "concept": "fiscal_year_end",
        "label": "Fiscal year end",
        "rule": "The fiscal year end is on the cover page; many retailers close on "
        "the Saturday nearest a month-end rather than December 31.",
    },
    {
        "concept": "effective_tax_rate",
        "label": "Effective tax rate",
        "rule": "The effective tax rate comes from the income-tax footnote "
        "(provision / pre-tax income); it differs from the 21% federal statutory rate.",
    },
]

# Per company: the correct value for each field plus the look-alike distractors
# that actually appear elsewhere on the same page.
COMPANIES: Dict[str, Dict[str, object]] = {
    "Northwind Retail Corp": {
        "fiscal_year": "FY2024",
        "values": {
            "total_revenue": ("$391,035", ["$365,817", "$201,183", "$94,930"]),
            "net_income": ("$24,160", ["$31,510", "$28,213", "$5,140"]),
            "total_assets": ("$252,200", ["$76,877", "$135,726", "$45,690"]),
            "diluted_eps": ("$2.41", ["$2.45", "$1.98", "$3.10"]),
            "fiscal_year_end": ("Feb 1, 2025", ["Dec 31, 2024", "Jan 31, 2025", "Feb 3, 2024"]),
            "effective_tax_rate": ("14.7%", ["21.0%", "12.3%", "19.5%"]),
        },
    },
    "Cascade Logistics Inc": {
        "fiscal_year": "FY2024",
        "values": {
            "total_revenue": ("$88,432", ["$84,002", "$41,205", "$22,118"]),
            "net_income": ("$3,902", ["$6,114", "$4,880", "$1,205"]),
            "total_assets": ("$71,540", ["$28,330", "$39,902", "$12,007"]),
            "diluted_eps": ("$4.12", ["$4.20", "$3.55", "$5.01"]),
            "fiscal_year_end": ("Dec 31, 2024", ["Jan 28, 2025", "Dec 26, 2024", "Dec 31, 2023"]),
            "effective_tax_rate": ("22.8%", ["21.0%", "25.1%", "18.0%"]),
        },
    },
    "Meridian Software Inc": {
        "fiscal_year": "FY2024",
        "values": {
            "total_revenue": ("$61,271", ["$56,189", "$29,540", "$15,002"]),
            "net_income": ("$18,344", ["$21,002", "$19,870", "$4,221"]),
            "total_assets": ("$132,880", ["$48,210", "$70,115", "$22,640"]),
            "diluted_eps": ("$7.66", ["$7.80", "$6.95", "$8.40"]),
            "fiscal_year_end": ("Jun 30, 2024", ["Dec 31, 2024", "Jul 1, 2024", "Jun 30, 2023"]),
            "effective_tax_rate": ("16.2%", ["21.0%", "13.9%", "20.4%"]),
        },
    },
}


def _letter(options: List[str], value: str) -> str:
    return f"{chr(65 + options.index(value))}) {value}"


def _value(prediction: str) -> str:
    """Strip an 'A) ' option prefix to recover the bare extracted value."""
    return prediction.split(")", 1)[-1].strip() if ")" in prediction else prediction.strip()


def _options(sample_id: str, correct: str, distractors: List[str]) -> List[str]:
    """Deterministically order the candidate values (correct slot varies per id)."""
    opts = [correct, *distractors]
    h = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
    rot = h % len(opts)
    return opts[rot:] + opts[:rot]


def _excerpt(company: str, fy: str, field: Dict[str, str], options: List[str]) -> str:
    lines = [
        f"FORM 10-K excerpt — {company} ({fy})",
        f"Extract: {field['label']}",
        "Candidate values found on the page:",
    ]
    lines += [f"  {chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


def _evaluate(prediction: str, sample: Sample) -> bool:
    return _value(prediction) == _value(sample.answer)


def build_10k_task(companies: List[str], name: str = "10k-extraction") -> Task:
    """One Sample per (company, field): extract the right value from the page."""
    samples: List[Sample] = []
    for company in companies:
        info = COMPANIES[company]
        fy = str(info["fiscal_year"])
        values = info["values"]  # type: ignore[assignment]
        for field in FIELDS:
            concept = field["concept"]
            correct, distractors = values[concept]  # type: ignore[index]
            sid = f"{company.split()[0].lower()}-{concept}"
            options = _options(sid, correct, distractors)
            samples.append(
                Sample(
                    id=sid,
                    question=_excerpt(company, fy, field, options),
                    answer=_letter(options, correct),
                    concept=concept,  # the extraction rule key ACE will learn
                    rule_text=field["rule"],
                    metadata={
                        "options": options,
                        "company": company,
                        "fiscal_year": fy,
                        "field": field["label"],
                    },
                )
            )
    return Task(
        name=name,
        samples=samples,
        evaluate=_evaluate,
        description="Extract structured financials from 10-K pages into a CSV.",
    )


def _write_csv(rows: List[Tuple[str, str, str, str, str, bool]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["company", "fiscal_year", "field", "extracted_value", "expected_value", "correct"]
        )
        w.writerows(rows)


def run(csv_path: str | None = None) -> dict:
    cleanup = False
    if csv_path is None:
        fd, csv_path = tempfile.mkstemp(suffix="_10k_extractions.csv")
        os.close(fd)
        cleanup = True

    # A base agent that innately knows only some of the line-item rules.
    env = TeachingEnvironment(known_fraction=0.25, seed=11)

    # Learn from two companies' filings; extract from a brand-new, held-out one.
    train_companies = ["Northwind Retail Corp", "Cascade Logistics Inc"]
    held_out = "Meridian Software Inc"
    train = build_10k_task(train_companies, name="10k-train")
    test = build_10k_task([held_out], name="10k-test")
    by_id = {s.id: s for s in test.samples}

    # 1) Baseline: extract from the unseen filing with no learned playbook.
    base = StaticAgent(SimulatedLLM(env)).run(test)

    # 2) ACE: learn extraction rules offline from labeled feedback, then extract.
    ace = ACE(SimulatedLLM(env), ACEConfig(epochs=4))
    ace.adapt_offline(train)
    adapted = ace.evaluate(test)

    # 3) Port the extractions to CSV — one row per (company, field).
    rows: List[Tuple[str, str, str, str, str, bool]] = []
    for rec in adapted.history:
        s = by_id[rec.sample_id]
        rows.append(
            (
                str(s.metadata["company"]),
                str(s.metadata["fiscal_year"]),
                str(s.metadata["field"]),
                _value(rec.prediction),
                _value(rec.ground_truth or ""),
                bool(rec.correct),
            )
        )
    rows.sort(key=lambda r: (r[0], r[2]))
    _write_csv(rows, csv_path)

    learned = [b.content for b in ace.playbook]
    result = {
        "held_out_company": held_out,
        "base_accuracy": base.accuracy,
        "ace_accuracy": adapted.accuracy,
        "playbook_bullets": len(ace.playbook),
        "rules_learned": learned,
        "rows_written": len(rows),
        "csv_path": csv_path,
        "csv_header": [
            "company",
            "fiscal_year",
            "field",
            "extracted_value",
            "expected_value",
            "correct",
        ],
        "sample_rows": rows[:3],
    }
    if cleanup:
        os.remove(csv_path)
    return result


def main() -> int:
    r = run()
    print("10-K extraction agent, optimized by ACE")
    print("=" * 44)
    print(f"Held-out filing extracted         : {r['held_out_company']}")
    print(f"Extraction accuracy (no learning) : {r['base_accuracy']:.1f}%")
    print(f"Extraction accuracy (+ ACE)       : {r['ace_accuracy']:.1f}%")
    print(f"Extraction rules learned          : {r['playbook_bullets']}")
    print()
    print("The agent wrote itself these extraction rules:")
    for rule in r["rules_learned"]:
        print(f"  • {rule}")
    print()
    print(f"Ported {r['rows_written']} extractions to CSV → {r['csv_path']}")
    print("First rows:")
    print("  " + ", ".join(r["csv_header"]))
    for row in r["sample_rows"]:
        print("  " + ", ".join(str(c) for c in row))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
