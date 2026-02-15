"""
Command-Line Interface for Job Application Summarizer.

This module provides the CLI commands using Click:
- process: Analyze applicant documents
- serve: Launch web viewer
- export: Generate CSV summary
- status: Show processing status

Usage:
    python run.py <command> [options]
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import config

# Rich console for pretty output
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    Job Application Summarizer - Process academic job applications with LLMs.
    
    Process applications, evaluate candidates, and browse results in a web interface.
    """
    pass


@main.command()
@click.argument("applicants_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model", "-m", default=None, help="Override LLM model for all tasks")
@click.option("--force", "-f", is_flag=True, help="Reprocess even if results exist")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without doing it")
def process(applicants_dir: Path, model: str, force: bool, dry_run: bool):
    """
    Process applicant documents and generate evaluations.
    
    APPLICANTS_DIR should contain subdirectories for each applicant,
    with their application documents (PDF/DOCX) inside.
    
    Example:
        python run.py process ../vap-search-2025/
    """
    console.print(f"[bold blue]Processing applications from:[/] {applicants_dir}")
    
    # TODO: Implement processing pipeline
    # 1. Discover applicant directories
    # 2. For each applicant:
    #    a. Extract text from documents
    #    b. Categorize documents
    #    c. Build profile
    #    d. Evaluate candidate
    #    e. Save results to data/results/{name}/
    
    # Placeholder: list what we found
    applicants = [d for d in applicants_dir.iterdir() if d.is_dir()]
    
    if dry_run:
        console.print(f"[yellow]Dry run - would process {len(applicants)} applicants:[/]")
        for a in applicants:
            docs = list(a.glob("*.pdf")) + list(a.glob("*.docx")) + list(a.glob("*.doc"))
            console.print(f"  • {a.name}: {len(docs)} documents")
        return
    
    console.print(f"[yellow]Found {len(applicants)} applicants - processing not yet implemented[/]")
    console.print("[dim]TODO: Implement extraction → categorization → evaluation pipeline[/]")


@main.command()
@click.option("--host", "-h", default=None, help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", "-p", default=None, type=int, help="Port to bind to (default: 5000)")
@click.option("--debug/--no-debug", default=True, help="Enable debug mode")
def serve(host: str, port: int, debug: bool):
    """
    Launch the web viewer for browsing applicants.
    
    Starts a local Flask server to view processed applications.
    
    Example:
        python run.py serve
        python run.py serve --port 8080
    """
    from src.viewer.app import create_app
    
    app = create_app()
    
    # Use provided values or fall back to config
    host = host or config.viewer.host
    port = port or config.viewer.port
    
    console.print(f"[bold green]Starting viewer at http://{host}:{port}[/]")
    console.print("[dim]Press Ctrl+C to stop[/]")
    
    app.run(host=host, port=port, debug=debug)


@main.command()
@click.argument("applicants_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Output CSV file")
def export(applicants_dir: Path, output: Path):
    """
    Export applicant summaries to CSV for the search committee.
    
    Example:
        python run.py export ../vap-search-2025/ --output summary.csv
    """
    console.print(f"[bold blue]Exporting to:[/] {output}")
    
    # TODO: Implement CSV export
    # Read from data/results/, generate summary CSV
    
    console.print("[yellow]Export not yet implemented[/]")


@main.command()
@click.argument("applicants_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def status(applicants_dir: Path):
    """
    Show processing status for all applicants.
    
    Example:
        python run.py status ../vap-search-2025/
    """
    # TODO: Check data/results/ for each applicant, show status table
    
    table = Table(title="Processing Status")
    table.add_column("Applicant", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Teaching", justify="center")
    table.add_column("Research", justify="center")
    
    applicants = [d for d in applicants_dir.iterdir() if d.is_dir()]
    
    for a in applicants:
        docs = list(a.glob("*.pdf")) + list(a.glob("*.docx"))
        
        # Check if processed
        result_dir = config.results_dir / a.name
        if result_dir.exists():
            status = "[green]✓ Done[/]"
            teaching = "⭐⭐⭐"  # TODO: Read from results
            research = "⭐⭐"
        else:
            status = "[yellow]○ Pending[/]"
            teaching = "-"
            research = "-"
        
        table.add_row(a.name, str(len(docs)), status, teaching, research)
    
    console.print(table)


if __name__ == "__main__":
    main()
