"""
main.py
=======

THE FRONT DOOR — terminal entry point for the entire system.

TWO COMMANDS:

  python main.py run
      Build Qdrant index (skips if already built)
      Process all 56 tickets in support_tickets.csv
      Write predictions to support_tickets/output.csv
      Print run summary

  python main.py run --rebuild
      Same as above but forces a full Qdrant index rebuild first.
      Use when corpus or chunking logic has changed.

  python main.py eval
      Score output.csv against sample_support_tickets.csv
      Print F1, accuracy, false reply rate, over-escalation rate

DESIGN PRINCIPLE:
  main.py owns ZERO business logic.
  It reads files, calls other modules, writes files, prints output.
  All intelligence lives in agent.py, retriever.py, guardrails.py etc.
  This makes main.py easy to read and easy to test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Add the code/ directory to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import agent
import retriever
import tracer

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Load .env file from project root (GEMINI_API_KEY lives here)
# load_dotenv() is idempotent — safe to call multiple times
load_dotenv(Path(__file__).parent.parent / ".env")

# Rich console — gives us coloured terminal output for free
# WHY Rich over plain print?
#   Panels, tables, colours make the output readable during a demo or judge run.
#   Zero extra complexity — just swap print() for console.print().
console = Console()

# Typer app — turns functions into CLI commands
# WHY Typer over argparse?
#   Typer uses type hints to infer argument types automatically.
#   Less boilerplate, cleaner code, free --help generation.
app = typer.Typer(
    name        = "orchestrate",
    help        = "HackerRank Orchestrate — Support Triage Agent",
    add_completion = False,   # disable shell completion (not needed for hackathon)
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — single source of truth
# All file paths defined once here, not scattered across files.
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR     = Path(__file__).parent.parent
TICKETS_PATH = ROOT_DIR / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH  = ROOT_DIR / "support_tickets" / "output.csv"
SAMPLE_PATH  = ROOT_DIR / "support_tickets" / "sample_support_tickets.csv"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _check_api_key() -> bool:
    """
    Check GEMINI_API_KEY is set. Print a clear error if not.

    WHY check here and not inside agent.py?
        Fail fast — better to catch a missing key before loading models,
        building the index, and reading CSVs. Saves the user 30 seconds
        of waiting before hitting an error.

    Returns True if key exists, False if missing.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        console.print(Panel(
            "[red]ERROR: GEMINI_API_KEY not found in environment.[/red]\n\n"
            "Create a [bold].env[/bold] file in the project root:\n"
            "  [green]GEMINI_API_KEY=your_key_here[/green]\n\n"
            "Get your key at: [blue]https://aistudio.google.com/apikey[/blue]",
            title="Missing API Key",
            border_style="red",
        ))
        return False
    return True


def _print_header() -> None:
    """Print the startup banner."""
    console.print(Panel(
        "[bold cyan]HackerRank Orchestrate[/bold cyan]\n"
        "Agentic RAG Support Triage Agent\n"
        "[dim]Dense + Sparse hybrid search · Cross-encoder re-ranking · 4-layer guardrails[/dim]",
        border_style="cyan",
    ))


def _load_tickets() -> pd.DataFrame | None:
    """
    Load support_tickets.csv into a DataFrame.

    WHY check file existence here?
        Clear error message beats a cryptic pandas FileNotFoundError.

    WHY keep_default_na=False?
        Pandas converts empty strings to NaN by default.
        "None" in the Company column would become NaN.
        We want to keep "None" as the string "None" so
        COMPANY_TO_DOMAIN mapping works correctly.

    Returns DataFrame or None if file not found.
    """
    if not TICKETS_PATH.exists():
        console.print(f"[red]ERROR: Tickets file not found: {TICKETS_PATH}[/red]")
        return None

    df = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    console.print(f"  [green]✓[/green] Loaded [bold]{len(df)}[/bold] tickets from {TICKETS_PATH.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND 1 — run
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def run(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Force rebuild of the Qdrant index from scratch. Use when corpus changes.",
    ),
) -> None:
    """
    Process all tickets in support_tickets.csv and write output.csv.

    WHAT HAPPENS:
        1. Check API key
        2. Build/load Qdrant index (hybrid dense + sparse vectors)
        3. Load support_tickets.csv
        4. Create trace file path
        5. Run agent over all tickets
        6. Write output.csv
        7. Print run summary

    The Qdrant index persists to disk (qdrant_db/).
    On subsequent runs, step 2 completes in <1 second.
    Use --rebuild to force a fresh index.
    """
    _print_header()

    # ── Step 1: Check API key ─────────────────────────────────────────────────
    if not _check_api_key():
        raise typer.Exit(code=1)

    # ── Step 2: Build / load Qdrant index ────────────────────────────────────
    # build_index() checks if collection already exists.
    # If yes and rebuild=False → returns immediately (fast startup).
    # If rebuild=True → deletes old collection and rebuilds from scratch.
    console.print("\n[bold]Step 1/3[/bold] — Loading models and connecting to Qdrant...")
    console.print(f"  Index rebuild: [{'yellow' if rebuild else 'dim'}]{'YES' if rebuild else 'NO (using existing index)'}[/{'yellow' if rebuild else 'dim'}]")

    try:
        qdrant_client = retriever.build_index(force_rebuild=rebuild)
    except Exception as e:
        console.print(f"[red]ERROR: Failed to build index: {e}[/red]")
        raise typer.Exit(code=1)

    # ── Step 3: Load tickets ──────────────────────────────────────────────────
    console.print("\n[bold]Step 2/3[/bold] — Reading tickets...")
    tickets_df = _load_tickets()
    if tickets_df is None:
        raise typer.Exit(code=1)

    # ── Step 4: Create trace file ─────────────────────────────────────────────
    # One trace file per run — named by timestamp
    trace_path = tracer.make_trace_path()
    console.print(f"  [green]✓[/green] Trace file: {trace_path.name}")

    # ── Step 5: Run agent ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 3/3[/bold] — Running agent over {len(tickets_df)} tickets...")
    console.print(f"  [dim]Rate limit: ~4s sleep between tickets · Est. time: ~{len(tickets_df) * 6 // 60} min[/dim]\n")
    console.print("─" * 55)

    try:
        output_df = agent.run_agent(
            tickets_df = tickets_df,
            client     = qdrant_client,
            trace_path = trace_path,
        )
    except KeyboardInterrupt:
        # User pressed Ctrl+C mid-run
        # Write whatever we have so far — partial output is better than nothing
        console.print("\n[yellow]Run interrupted by user. Writing partial output...[/yellow]")
        # output_df may be incomplete here — handled below

    # ── Step 6: Write output.csv ──────────────────────────────────────────────
    # Column order matches the problem statement schema exactly
    output_columns = ["status", "product_area", "response", "justification", "request_type"]

    # Ensure all columns exist (defensive — post_generation should guarantee this)
    for col in output_columns:
        if col not in output_df.columns:
            output_df[col] = ""

    output_df[output_columns].to_csv(OUTPUT_PATH, index=False)

    console.print("─" * 55)
    console.print(f"\n  [bold green]✓ Output written to {OUTPUT_PATH}[/bold green]")

    # ── Step 7: Print run summary ─────────────────────────────────────────────
    tracer.print_run_summary(trace_path)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND 2 — eval
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def eval() -> None:
    """
    Score output.csv against sample_support_tickets.csv.

    WHAT IT SCORES:
        status F1          — replied vs escalated classification accuracy
        request_type F1    — 4-class classification (product_issue, bug, etc.)
        product_area acc   — how often we got the right support category
        false reply rate   — escalate-worthy tickets we answered (dangerous)
        over-escalation    — simple FAQs we escalated (wasteful)

    Run `python main.py run` first to generate output.csv.
    """
    _print_header()

    # ── Check output.csv exists ───────────────────────────────────────────────
    if not OUTPUT_PATH.exists():
        console.print(Panel(
            "[red]output.csv not found.[/red]\n\n"
            "Run the agent first:\n"
            "  [green]python main.py run[/green]",
            title="No Predictions Found",
            border_style="red",
        ))
        raise typer.Exit(code=1)

    # ── Check sample CSV exists ───────────────────────────────────────────────
    if not SAMPLE_PATH.exists():
        console.print(f"[red]ERROR: Sample file not found: {SAMPLE_PATH}[/red]")
        raise typer.Exit(code=1)

    # ── Load both CSVs ────────────────────────────────────────────────────────
    output_df = pd.read_csv(OUTPUT_PATH,  keep_default_na=False)
    sample_df = pd.read_csv(SAMPLE_PATH,  keep_default_na=False)

    console.print(f"  Predictions : {len(output_df)} rows")
    console.print(f"  Sample data : {len(sample_df)} rows\n")

    # ── Import and run evaluator ──────────────────────────────────────────────
    # WHY import here and not at top?
    #   evaluator.py imports sklearn which adds ~1s to startup.
    #   We only pay that cost when eval is actually called.
    #   The run command doesn't need sklearn at all.
    try:
        import evaluator
        metrics = evaluator.evaluate(
            predictions = output_df,
            ground_truth = sample_df,
        )
    except ImportError:
        console.print("[red]ERROR: evaluator.py not found or missing dependencies.[/red]")
        console.print("Install: [green]pip install scikit-learn rouge-score[/green]")
        raise typer.Exit(code=1)

    # ── Print results table ───────────────────────────────────────────────────
    table = Table(title="Evaluation Results", border_style="cyan")
    table.add_column("Metric",  style="bold white", min_width=25)
    table.add_column("Score",   style="bold green",  min_width=10)
    table.add_column("Notes",   style="dim",          min_width=30)

    # Helper to colour score red/yellow/green based on thresholds
    def score_colour(value: float, good: float = 0.8, ok: float = 0.6) -> str:
        colour = "green" if value >= good else ("yellow" if value >= ok else "red")
        return f"[{colour}]{value:.3f}[/{colour}]"

    for metric_name, metric_value, note in metrics:
        table.add_row(
            metric_name,
            score_colour(metric_value),
            note,
        )

    console.print(table)

    # ── Also print trace stats if latest trace exists ─────────────────────────
    latest_trace = tracer.get_latest_trace_path()
    if latest_trace:
        tracer.print_run_summary(latest_trace)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # WHY check __name__ == "__main__"?
    #   Standard Python pattern — ensures app() only runs when this file
    #   is executed directly, not when imported as a module.
    #   Prevents side effects if another file does `import main`.
    app()