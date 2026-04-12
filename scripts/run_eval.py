#!/usr/bin/env python3
"""CLI entry point for SafeEval pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


@click.command()
@click.option("--models", "-m", multiple=True, default=["gpt-4o"],
              help="Model names to evaluate (repeatable)")
@click.option("--benchmarks/--no-benchmarks", default=True, help="Run capability benchmarks")
@click.option("--attacks/--no-attacks", default=True, help="Run red-team attack sets")
@click.option("--num-bench", default=10, help="Samples per benchmark")
@click.option("--num-attack", default=10, help="Samples per attack set")
@click.option("--no-judge", is_flag=True, default=False, help="Skip LLM-as-judge (use mock scores)")
@click.option("--output-dir", default="data/results", help="Output directory for results")
@click.option("--openai-key", envvar="OPENAI_API_KEY", default=None)
@click.option("--anthropic-key", envvar="ANTHROPIC_API_KEY", default=None)
def main(models, benchmarks, attacks, num_bench, num_attack, no_judge, output_dir, openai_key, anthropic_key):
    """Run the SafeEval four-layer evaluation pipeline."""
    from safeeval.pipeline import SafeEvalPipeline, RunConfig

    console.print("[bold blue]SafeEval[/bold blue] — four-layer LLM evaluation framework")
    console.print(f"Models: {list(models)}")
    console.print(f"Benchmarks: {benchmarks}, Attacks: {attacks}, Judge: {not no_judge}")

    cfg = RunConfig(
        models=list(models),
        run_benchmarks=benchmarks,
        run_attacks=attacks,
        num_benchmark_samples=num_bench,
        num_attack_samples=num_attack,
        use_judge=not no_judge,
        output_dir=output_dir,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
    )

    pipeline = SafeEvalPipeline(cfg)

    with console.status("[bold green]Running evaluation..."):
        df = pipeline.run()

    # Print summary
    console.print(f"\n[green]✓[/green] Completed {len(df):,} evaluations")

    table = Table(title="Summary", show_header=True)
    table.add_column("Model")
    table.add_column("Inferences", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("ASR", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Avg Cost/1K", justify="right")

    for model in df["model_name"].unique():
        mdf = df[df["model_name"] == model]
        attacks_df = mdf[mdf["is_attack"]]
        pass_rate = round(mdf["overall_pass"].mean() * 100, 1)
        asr = round(attacks_df["harmfulness_score"].gt(0.5).mean() * 100, 1) if len(attacks_df) > 0 else 0.0
        avg_lat = int(mdf["latency_ms"].mean())
        avg_cost = round(mdf["cost_per_1k"].mean(), 3)
        table.add_row(model, str(len(mdf)), f"{pass_rate}%", f"{asr}%",
                      f"{avg_lat}ms", f"${avg_cost}")

    console.print(table)
    console.print(f"\nResults saved to [bold]{output_dir}[/bold]")
    console.print("Launch dashboard: [bold]streamlit run dashboard/app.py[/bold]")


if __name__ == "__main__":
    main()
