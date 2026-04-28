from typing import Annotated

import typer
from rich.console import Console

from eval.runner import run_evaluation
from src.agent.graph import run_graph

app = typer.Typer(
    name="rag",
    help="Hybrid RAG system for master's thesis research",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    question: str | None = typer.Argument(None, help="Question to ask (omit for interactive mode)"),
    provider: str | None = typer.Option(None, help="LLM provider (openai, gemini, ollama, groq, openrouter)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    ablation: str = typer.Option(
        "full", "--ablation", "-a", help="Ablation mode: full, no-verifier, no-synthesizer, poc"
    ),
):
    """Chat with the system. Pass a question or start interactive mode."""
    _configure(provider, verbose)
    from src.agent.ablation import AblationMode

    mode = AblationMode(ablation)

    if question:
        output = run_graph(question, mode=mode)
        console.print(f"\n[bold green]Answer:[/bold green]\n{output.get('final_answer', '')}")
        _print_metrics(output)
    else:
        console.print(f"[bold]Interactive mode ({mode.value}).[/bold] Type 'exit' to quit.\n")
        while True:
            q = console.input(f"[bold cyan][{mode.value}] Question:[/bold cyan] ")
            if q.strip().lower() in ("exit", "quit", "q"):
                break
            output = run_graph(q, mode=mode)
            console.print(f"\n[bold green]Answer:[/bold green]\n{output.get('final_answer', '')}")
            _print_metrics(output)
            console.print()


@app.command()
def ingest(
    provider: str | None = typer.Option(None, help="LLM provider for embeddings"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ingest structured (CSV) and unstructured (PDF) data."""
    _configure(provider, verbose)
    from src.services.ingest_service import ingest_structured, ingest_unstructured

    console.rule("[bold]Ingesting Structured Data[/bold]")
    ingest_structured()

    console.rule("[bold]Ingesting Unstructured Data[/bold]")
    ingest_unstructured()


@app.command(name="semantic-map")
def semantic_map(
    provider: str | None = typer.Option(None, help="LLM provider"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Auto-generate semantic maps from ingested data."""
    _configure(provider, verbose)
    from scripts.generate_semantic_map import generate_structured_map, generate_unstructured_map

    console.rule("[bold]Generating Structured Semantic Map[/bold]")
    generate_structured_map()

    console.rule("[bold]Generating Unstructured Semantic Map[/bold]")
    generate_unstructured_map()


@app.command(name="eval")
def evaluate(
    provider: str | None = typer.Option(None, help="LLM provider (openai, gemini, ollama, groq, openrouter)"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Number of parallel workers"),
    run_id: str | None = typer.Option(None, "--run-id", help="Label for this run"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
    snapshot: str | None = typer.Option(None, help="Path to snapshot file for re-evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    ablation: str = typer.Option(
        "full", "--ablation", "-a", help="Ablation mode: full, no-verifier, no-synthesizer, poc"
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Diretório onde snapshots/results/checkpoint serão gravados. "
             "Default: data/outputs/. Use para isolar experimentos (ex.: data/outputs/_mix/gpt-LSS).",
    ),
):
    """Run the evaluation suite.

    Para rodar com modelos diferentes por node do grafo, defina env vars no
    shell antes do comando. Chaves: PLANNER, SQL, SYNTHESIS, VERIFIER, ROUTER.
    Cada uma aceita `<KEY>_PROVIDER` e `<KEY>_MODEL`.

    Exemplo (mix GPT: Planner grande, SQL/Synth pequenos):
        PLANNER_PROVIDER=openai PLANNER_MODEL=gpt-5 \\
        SQL_PROVIDER=openai SQL_MODEL=gpt-5-nano \\
        SYNTHESIS_PROVIDER=openai SYNTHESIS_MODEL=gpt-5-nano \\
        OPENAI_MODEL=gpt-5 \\
        rag eval --ablation no-synthesizer --run-id mix-gpt-LSS \\
            --output-dir data/outputs/_mix/gpt-LSS -c 3
    """
    from pathlib import Path

    _configure(provider, verbose)
    from eval.runner import _evaluate_output, save_results
    from src.agent.ablation import AblationMode

    mode = AblationMode(ablation)
    out_dir = Path(output_dir) if output_dir else None

    if snapshot:
        import json

        console.print(f"[bold]Re-evaluating from snapshot: {snapshot}[/bold]")
        with open(snapshot, encoding="utf-8") as f:
            snapshots = json.load(f)

        results = [_evaluate_output(s["output"], s["input"]) for s in snapshots]
        save_results(results, snapshots, run_id or "re-eval", ablation_mode=mode, output_dir=out_dir)
    else:
        results, snapshots, effective_run_id = run_evaluation(
            concurrency=concurrency,
            run_id=run_id,
            resume=resume,
            ablation_mode=mode,
            output_dir=out_dir,
        )
        if results:
            results_path = save_results(
                results, snapshots, effective_run_id, ablation_mode=mode, output_dir=out_dir
            )

            from eval.report import display_report, load_results

            display_report(load_results(results_path))


@app.command(name="report")
def report_cmd(
    files: Annotated[list[str], typer.Argument(help="Path(s) to results JSON file(s)")],
):
    """Display results in the terminal."""
    from pathlib import Path

    from eval.report import display_comparative_report, display_report, load_results

    datasets = [load_results(Path(f)) for f in files]
    if len(datasets) == 1:
        display_report(datasets[0])
    else:
        display_comparative_report(datasets)


@app.command(name="rejudge")
def rejudge_cmd(
    outputs_dir: str = typer.Option(
        "data/outputs", "--outputs-dir", "-o", help="Diretório com subpastas de snapshots"
    ),
    snapshot: str | None = typer.Option(
        None, "--snapshot", "-s", help="Caminho específico de um snapshot_*.json (ignora outputs-dir)"
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Processar apenas os primeiros N snapshots (útil para teste)"
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", "-c", help="Itens paralelos por snapshot (>=1)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Reportar apenas, sem gravar arquivos nem chamar LLMs"),
    migrate_only: bool = typer.Option(False, "--migrate-only", help="Apenas migra formato legado para o novo, sem chamar J2/J3"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Re-julgamento idempotente usando os juízes configurados.

    Verifica cada item: se já foi julgado por J1/J2/J3, pula. Caso contrário, chama
    o juiz faltante e atualiza snapshot + results. Seguro rodar múltiplas vezes.
    Gravação incremental após cada juiz — cancelamento não perde trabalho.

    Exemplos:
      rag rejudge --migrate-only                     # migra formato, sem custo
      rag rejudge --snapshot data/outputs/gpt/snap_X.json  # 1 arquivo
      rag rejudge --limit 1                          # só o 1º snapshot
      rag rejudge -c 5                               # 5 itens em paralelo
      rag rejudge                                    # tudo, sequencial
    """
    from pathlib import Path

    from eval.judges import rejudge_snapshots

    _configure(None, verbose)

    snapshot_paths = [Path(snapshot)] if snapshot else None
    outputs_path = None if snapshot else Path(outputs_dir)

    stats = rejudge_snapshots(
        outputs_dir=outputs_path,
        snapshot_paths=snapshot_paths,
        limit=limit,
        concurrency=concurrency,
        dry_run=dry_run,
        migrate_only=migrate_only,
    )
    console.print(f"[green]Re-julgamento concluído:[/green] {stats}")


@app.command(name="status")
def status_cmd(
    outputs_dir: str = typer.Option(
        "data/outputs", "--outputs-dir", "-o", help="Diretório raiz com subpastas de modelos"
    ),
    show_complete: bool = typer.Option(
        False, "--all", help="Mostra todos os snapshots (padrão: só incompletos)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Reporta status de julgamento (j1/j2/j3) de todos os snapshots.

    Por padrão mostra só os snapshots com itens faltantes. Use --all para ver tudo.
    """
    from pathlib import Path

    from rich.table import Table

    from eval.judges import check_judgement_status

    _configure(None, verbose)
    rows = check_judgement_status(Path(outputs_dir))

    table = Table(title="Status dos julgamentos por snapshot", show_lines=False)
    table.add_column("Modelo", style="cyan")
    table.add_column("Arquitetura", style="dim")
    table.add_column("Itens", justify="right")
    table.add_column("J1", justify="right")
    table.add_column("J2", justify="right")
    table.add_column("J3", justify="right")
    table.add_column("Completos", justify="right")
    table.add_column("Faltantes (ids)", style="yellow")

    filtered = [r for r in rows if show_complete or r["complete"] < r["total"]]
    incomplete_total = 0

    for r in filtered:
        faltantes = []
        for j, ids in r["missing_per_judge"].items():
            if ids:
                faltantes.append(f"{j}: {','.join(ids[:5])}" + ("..." if len(ids) > 5 else ""))
        missing_str = " | ".join(faltantes) or "-"
        complete_style = "[green]" if r["complete"] == r["total"] else "[red]"
        table.add_row(
            r["model"],
            r["arch"],
            str(r["total"]),
            str(r["by_judge"]["j1"]),
            str(r["by_judge"]["j2"]),
            str(r["by_judge"]["j3"]),
            f"{complete_style}{r['complete']}/{r['total']}",
            missing_str,
        )
        if r["complete"] < r["total"]:
            incomplete_total += 1

    console.print(table)
    total_snaps = len(rows)
    console.print(
        f"\n[bold]Resumo:[/bold] {total_snaps - incomplete_total}/{total_snaps} snapshots completos. "
        f"{incomplete_total} com itens faltantes."
    )


@app.command(name="exp-verifier-signal")
def exp_verifier_signal_cmd(
    outputs_dir: str = typer.Option(
        "data/outputs", "--outputs-dir", "-o", help="Diretório raiz com subpastas de modelos"
    ),
    model_key: list[str] | None = typer.Option(
        None,
        "--model-key",
        "-m",
        help="Pasta de modelo a processar. Pode ser repetido. Default: todas.",
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", "-c", help="Itens paralelos por snapshot (>=1). Threads."
    ),
    force: bool = typer.Option(
        False, "--force", help="Reprocessa mesmo itens já presentes no arquivo de sinais"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Experimento — Verifier signal-only sobre snapshots no-verifier existentes.

    Invoca o Verifier UMA única vez (sem laço de retry) sobre cada resposta já
    produzida pela arquitetura no-verifier. Mede utilidade do sinal como indicador
    de qualidade, sem custo de retry.

    PERSISTÊNCIA: cada snapshot `snapshot_X.json` gera um arquivo paralelo
    `snapshot_X.verifier_signal.json` no mesmo diretório contendo {item_id: features}.
    Snapshots originais não são modificados.

    GRAVAÇÃO INCREMENTAL: após cada item. Cancelar não perde trabalho em andamento.

    IDEMPOTÊNCIA: itens já presentes no arquivo de sinais são pulados (use --force
    para refazer).

    Exemplos:
        rag exp-verifier-signal                        # todos os modelos
        rag exp-verifier-signal --model-key gemini-pro # uma pasta
        rag exp-verifier-signal -m gpt -m gpt-nano -c 5  # 2 modelos, 5 paralelos
    """
    from pathlib import Path

    from eval.experiments.verifier_signal import run_on_outputs_dir

    _configure(None, verbose)
    totals = run_on_outputs_dir(
        Path(outputs_dir),
        model_keys=list(model_key) if model_key else None,
        concurrency=concurrency,
        force=force,
    )
    console.print(
        f"\n[bold green]Experimento concluído:[/bold green] "
        f"{totals['snapshots']} snapshots, "
        f"{totals['n_processed']} itens processados, "
        f"{totals['n_skipped']} pulados, "
        f"{totals['n_failed']} falharam."
    )


@app.command(name="analyze")
def analyze_cmd(
    outputs_dir: str = typer.Option(
        "data/outputs", "--outputs-dir", "-o", help="Diretório raiz com subpastas de modelos"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Gera os artefatos de análise: CSV agregado, testes estatísticos e markdown qualitativo.

    Saídas em `<outputs-dir>/_analysis/`:
      - aggregated.csv, aggregated_runs.csv
      - stats.json, stats_summary.txt
      - qualitative_samples.md
    """
    from pathlib import Path

    _configure(None, verbose)
    outputs = Path(outputs_dir)
    analysis_dir = outputs / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from scripts import aggregate, qualitative, stats

    console.rule("[bold]1/3 Agregação[/bold]")
    aggregate.run(outputs, analysis_dir)

    console.rule("[bold]2/3 Testes estatísticos[/bold]")
    stats.run(analysis_dir, outputs)

    console.rule("[bold]3/3 Mineração qualitativa[/bold]")
    qualitative.run(outputs, analysis_dir)

    console.print(f"\n[bold green]Análise concluída em:[/bold green] {analysis_dir}")


@app.command()
def graph(
    ablation: str = typer.Option("full", "--ablation", "-a", help="Ablation mode to display"),
):
    """Display the agent graph structure (Mermaid)."""
    from src.agent.ablation import AblationMode
    from src.agent.graph import get_graph_mermaid

    mode = AblationMode(ablation)
    mermaid = get_graph_mermaid(mode)
    console.print(f"[bold]Graph Structure ({mode.value}) — Mermaid:[/bold]\n")
    console.print(mermaid)


def _print_metrics(output: dict) -> None:
    """Prints execution metrics from the graph output."""
    import time

    from rich.table import Table

    total_start = output.get("total_start", 0)
    total_time = round(time.perf_counter() - total_start, 2) if total_start else 0

    usage = output.get("token_usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    total_tokens = int(usage.get("total_tokens", 0))

    trace = output.get("trace", [])

    table = Table(title="Metrics", show_header=True, header_style="bold")
    table.add_column("Node", style="cyan")
    table.add_column("Provider", style="dim")
    table.add_column("Model", style="dim")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Tokens In", justify="right")
    table.add_column("Tokens Out", justify="right")
    table.add_column("Tokens Total", justify="right")

    for entry in trace:
        table.add_row(
            entry.get("node", ""),
            entry.get("provider", ""),
            entry.get("model", ""),
            str(entry.get("duration", "")),
            str(entry.get("input_tokens", "")),
            str(entry.get("output_tokens", "")),
            str(entry.get("total_tokens", "")),
        )

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        "",
        "",
        f"[bold]{total_time}s[/bold]",
        str(input_tokens),
        str(output_tokens),
        f"[bold]{total_tokens}[/bold]",
    )

    console.print()
    console.print(table)


def _configure(provider: str | None, verbose: bool) -> None:
    """Applies runtime configuration overrides."""
    import sys

    import loguru

    from src.config.settings import SETTINGS

    if provider:
        SETTINGS.PROVIDER = provider

    loguru.logger.remove()
    level = "DEBUG" if verbose else SETTINGS.LOG_LEVEL
    loguru.logger.add(sys.stderr, level=level)
