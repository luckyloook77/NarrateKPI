"""
NarrateKPI — End-to-End CLI Runner (Modules 1 + 2)

Pipeline:
  1. Run the Math Engine on every mock test case.
  2. Pass each ``summary_raw_json`` into the LLM Synthesis Engine.
  3. Print generated Markdown reports to the terminal.
  4. Write each report to ``reports_output/<account_id>.md``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

from llm_engine import generate_report
from math_engine import run_math_engine
from mock_data import ALL_CASES
from schemas import AnomalyReport


# ──────────────────────────────────────────────────────────────────────
#  ANSI colour helpers (Windows-safe)
# ──────────────────────────────────────────────────────────────────────

class Colours:
    """Minimal ANSI escape sequences; degrades gracefully on unsupported terminals."""

    _supports_colour: bool = sys.stdout.isatty()

    RESET = "\033[0m" if _supports_colour else ""
    BOLD = "\033[1m" if _supports_colour else ""
    RED = "\033[91m" if _supports_colour else ""
    GREEN = "\033[92m" if _supports_colour else ""
    YELLOW = "\033[93m" if _supports_colour else ""
    CYAN = "\033[96m" if _supports_colour else ""
    GREY = "\033[90m" if _supports_colour else ""
    MAGENTA = "\033[95m" if _supports_colour else ""


# ──────────────────────────────────────────────────────────────────────
#  Output helpers
# ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "reports_output"


def print_separator(char: str = "━", length: int = 72) -> None:
    print(char * length)


def write_report_file(account_id: str, markdown: str) -> str:
    """Write a Markdown report to disk and return the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = account_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(OUTPUT_DIR, f"{safe_name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return path


# ──────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full NarrateKPI pipeline: detect anomalies, then narrate them."""

    # ── Step 1: Run Math Engine & generate LLM reports ──────────────
    reports: List[AnomalyReport] = []
    report_cache: List[tuple[AnomalyReport, str]] = []  # (report, markdown)

    for idx, case in enumerate(ALL_CASES):
        report = run_math_engine(case)
        reports.append(report)

        summary = report.summary_raw_json
        client_name = summary.get("client_name", f"Case #{idx + 1}")
        account_id = summary.get("account_id", f"case_{idx + 1}")

        # Generate the narrative report (call the LLM / mock ONCE).
        markdown = generate_report(summary)
        report_cache.append((report, markdown))

    # ── Step 2: LLM-ready payload ───────────────────────────────────
    llm_payload = {
        "report_date": "2026-W29",
        "total_accounts_analyzed": len(reports),
        "accounts": [r.summary_raw_json for r in reports],
    }

    print(
        f"{Colours.BOLD}{Colours.MAGENTA}"
        f" NarrateKPI — Weekly Performance Report Pipeline "
        f"{Colours.RESET}"
    )
    print_separator("═")
    print(
        f" Accounts to analyse: {len(reports)}  |  "
        f"Mode: {'DRY-RUN' if not _has_api_key() else 'LIVE LLM'}"
    )
    print_separator("━")
    print()

    # ── Step 3: Write + preview ─────────────────────────────────────
    for idx, (report, markdown) in enumerate(report_cache):
        summary = report.summary_raw_json
        client_name = summary.get("client_name", f"Case #{idx + 1}")
        account_id = summary.get("account_id", f"case_{idx + 1}")

        print(
            f"{Colours.BOLD}{Colours.CYAN}"
            f"▶  Report for: {client_name}"
            f"{Colours.RESET}"
        )

        # Preview first ~20 lines.
        preview_lines = markdown.strip().split("\n")[:20]
        print(f"  {Colours.GREY}│{Colours.RESET}  " + f"\n  {Colours.GREY}│{Colours.RESET}  ".join(preview_lines))
        if len(markdown.strip().split("\n")) > 20:
            print(f"  {Colours.GREY}│{Colours.RESET}  {Colours.GREY}... (truncated preview){Colours.RESET}")
        print()

        # Write full report to file.
        path = write_report_file(account_id, markdown)
        print(
            f"  {Colours.GREEN}✓{Colours.RESET} Written → {Colours.BOLD}{path}{Colours.RESET}"
        )
        print_separator("─")
        print()

    # ── Step 4: Print full reports ─────────────────────────────────
    print(
        f"{Colours.BOLD}{Colours.MAGENTA}"
        f" Full Markdown Reports "
        f"{Colours.RESET}"
    )
    print_separator("═")

    for idx, (report, markdown) in enumerate(report_cache):
        client_name = report.summary_raw_json.get("client_name", f"Case #{idx + 1}")

        print(f"\n{Colours.BOLD}{Colours.CYAN}{'='*60}{Colours.RESET}")
        print(f"{Colours.BOLD}{Colours.CYAN}  Report: {client_name}{Colours.RESET}")
        print(f"{Colours.BOLD}{Colours.CYAN}{'='*60}{Colours.RESET}\n")
        print(markdown)
        print()

    # ── Step 5: Print LLM-ready JSON summary ───────────────────────
    print_separator("═")
    print(
        f"{Colours.BOLD}{Colours.CYAN}"
        f"LLM-Ready JSON Payload (for prompt-context reference)"
        f"{Colours.RESET}"
    )
    print_separator("═")
    print(json.dumps(llm_payload, indent=2, ensure_ascii=False))
    print_separator("═")

    # ── Summary ────────────────────────────────────────────────────
    print(
        f"{Colours.BOLD}{Colours.GREEN}"
        f"\n✓ Pipeline complete. {len(reports)} reports written to '{OUTPUT_DIR}/'."
        f"{Colours.RESET}"
    )


def _has_api_key() -> bool:
    """Check if any supported LLM API key is set in the environment."""
    for env_var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(env_var):
            return True
    return False


if __name__ == "__main__":
    main()
