"""
Simpsons Analysis Pipeline
==========================
Runs all analysis scripts in the correct dependency order:
  1. simpsons_cleaning.py        — data cleaning (must run first)
  2. question_1.py               — Q1 static ratings chart
  3. question_2.py               — Q2 viewership chart
  4. question_3.py               — Q3 correlation scatter
  5. question_4.py               — Q4 viewership by day
  6. question_5.py               — Q5 heatmap
  7. question_6.py               — Q6 production complexity
  8. q1_ratings_evolution.py     — Q1 interactive visualisations (3 charts)
  9. q2_viz1_interactive_filters.py — Q2 interactive filters

Usage
-----
    python run_pipeline.py [--scripts-dir PATH] [--stop-on-error]

Options
-------
  --scripts-dir   Directory containing the scripts (default: same folder as
                  this runner, or current working directory).
  --stop-on-error Abort the whole pipeline as soon as one script fails.
                  By default the runner continues and reports all failures
                  at the end.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ── Pipeline definition ────────────────────────────────────────────────────────
# Each entry is (script_filename, human_readable_description).
# Order matters: cleaning must precede all analysis scripts.

PIPELINE: list[tuple[str, str]] = [
    ("simpsons_cleaning.py",             "Data Cleaning"),
    ("question_1.py",                    "Q1 – Ratings Evolution (static)"),
    ("question_2.py",                    "Q2 – Viewership Evolution (static)"),
    ("question_3.py",                    "Q3 – Quality vs Popularity Scatter"),
    ("question_4.py",                    "Q4 – Viewership by Broadcast Day"),
    ("question_5.py",                    "Q5 – Season/Episode Heatmap"),
    ("question_6.py",                    "Q6 – Production Complexity Trends"),
    ("q1_ratings_evolution.py",          "Q1 – Interactive Ratings Visualisations"),
    ("q2_viz1_interactive_filters.py",   "Q2 – Interactive Filter Visualisation"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def banner(text: str) -> None:
    width = 66
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def step_header(index: int, total: int, name: str, desc: str) -> None:
    print(f"\n{BOLD}[{index}/{total}] {desc}{RESET}  {DIM}({name}){RESET}")
    print(f"{'·' * 50}")


def run_script(script_path: Path) -> tuple[bool, float, str]:
    """
    Execute *script_path* with the current Python interpreter.

    Returns
    -------
    (success, elapsed_seconds, combined_output)
    """
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=script_path.parent,   # run from the script's own directory so
                                  # relative paths (data/, outputs/) resolve
    )
    elapsed = time.perf_counter() - start
    combined = result.stdout + result.stderr
    return result.returncode == 0, elapsed, combined


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Simpsons analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scripts-dir",
        default=None,
        help=(
            "Directory that contains the pipeline scripts. "
            "Defaults to the folder of this runner file."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately when a script fails (default: keep going).",
    )
    args = parser.parse_args()

    # Resolve scripts directory
    if args.scripts_dir:
        scripts_dir = Path(args.scripts_dir).resolve()
    else:
        scripts_dir = Path(__file__).parent.resolve()

    banner("Simpsons Analysis Pipeline")
    print(f"  Scripts directory : {scripts_dir}")
    print(f"  Stop on error     : {args.stop_on_error}")
    print(f"  Steps             : {len(PIPELINE)}")

    total = len(PIPELINE)
    results: list[dict] = []

    pipeline_start = time.perf_counter()

    for idx, (filename, description) in enumerate(PIPELINE, start=1):
        script_path = scripts_dir / filename
        step_header(idx, total, filename, description)

        # ── Pre-flight: does the file exist? ───────────────────────────────
        if not script_path.exists():
            msg = f"Script not found: {script_path}"
            print(f"  {YELLOW}⚠  SKIPPED — {msg}{RESET}")
            results.append({"name": filename, "desc": description,
                             "status": "SKIPPED", "elapsed": 0.0, "reason": msg})
            continue

        # ── Execute ────────────────────────────────────────────────────────
        print(f"  Running…", end="", flush=True)
        success, elapsed, output = run_script(script_path)

        status_str  = f"{GREEN}✔  PASSED{RESET}" if success else f"{RED}✖  FAILED{RESET}"
        status_key  = "PASSED" if success else "FAILED"

        print(f"\r  {status_str}  ({fmt_time(elapsed)})")

        # Always show script output (trim if very long)
        if output.strip():
            lines = output.strip().splitlines()
            if len(lines) > 40:
                shown = lines[:20] + ["    ... (output truncated) ..."] + lines[-10:]
            else:
                shown = lines
            print(f"\n{DIM}" + "\n".join(f"    {l}" for l in shown) + RESET)

        results.append({
            "name":    filename,
            "desc":    description,
            "status":  status_key,
            "elapsed": elapsed,
            "reason":  "" if success else "Non-zero exit code",
        })

        if not success and args.stop_on_error:
            print(f"\n{RED}{BOLD}Pipeline aborted (--stop-on-error is set).{RESET}")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    pipeline_elapsed = time.perf_counter() - pipeline_start
    banner("Pipeline Summary")

    col_w = max(len(r["desc"]) for r in results) + 2
    header = f"  {'Step':<{col_w}}  {'Status':<8}  {'Time':>7}"
    print(header)
    print(f"  {'─'*col_w}  {'─'*8}  {'─'*7}")

    n_pass = n_fail = n_skip = 0
    for r in results:
        if r["status"] == "PASSED":
            colour, icon, n_pass = GREEN,  "✔", n_pass + 1
        elif r["status"] == "FAILED":
            colour, icon, n_fail = RED,    "✖", n_fail + 1
        else:
            colour, icon, n_skip = YELLOW, "⚠", n_skip + 1

        t = fmt_time(r["elapsed"]) if r["elapsed"] else "—"
        print(f"  {r['desc']:<{col_w}}  "
              f"{colour}{icon} {r['status']:<6}{RESET}  {t:>7}")
        if r["reason"]:
            print(f"  {' '*col_w}    {DIM}{r['reason']}{RESET}")

    print(f"\n  {'─'*50}")
    overall = f"{GREEN}ALL PASSED{RESET}" if n_fail == 0 and n_skip == 0 \
              else (f"{RED}{n_fail} FAILED{RESET}" if n_fail else "") \
              + (f"  {YELLOW}{n_skip} SKIPPED{RESET}" if n_skip else "")
    print(f"  Result : {overall or f'{GREEN}ALL PASSED{RESET}'}")
    print(f"  Passed : {n_pass}   Failed : {n_fail}   Skipped : {n_skip}")
    print(f"  Total time : {fmt_time(pipeline_elapsed)}\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
