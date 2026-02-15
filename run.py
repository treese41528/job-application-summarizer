#!/usr/bin/env python3
"""
Job Application Summarizer - Main Entry Point

Usage:
    python run.py process <applicants_dir> [--model MODEL] [--results-dir DIR] [--reprocess]
                          [--skip-rank] [--no-rag] [--no-ensemble] [--force-recreate-kbs]
                          [--position YAML]
    python run.py rank [--model MODEL] [--results-dir DIR] [--position YAML] [--ensemble]
    python run.py serve [--port PORT] [--results-dir DIR] [--applicants-dir DIR]
                        [--model MODEL] [--no-chat] [--position YAML]
    python run.py status <applicants_dir> [--results-dir DIR]
    python run.py export <applicants_dir> [--results-dir DIR] [--output FILE]
    python run.py cleanup-kbs [--results-dir DIR]

Examples:
    # Process with default config (VAP teaching at Purdue)
    python run.py process ../vap-search-2025/
    python run.py process ../vap-search-2025/ --model gemma3:12b
    python run.py process ../vap-search-2025/ --no-ensemble       # RAG only, single model
    python run.py process ../vap-search-2025/ --no-rag --no-ensemble  # Classic mode
    python run.py process ../vap-search-2025/ --force-recreate-kbs    # Rebuild all KBs
    python run.py process ../vap-search-2025/Alice_Smith/  # Single applicant

    # Process with a custom position profile
    python run.py process ../tt-search-2026/ --position positions/tt-research-2026.yaml
    python run.py process ../vap-search-2025/ --position positions/vap-teaching-2025.yaml

    # Ranking
    python run.py rank --model deepseek-r1:70b             # Re-rank with different model
    python run.py rank --ensemble                          # Multi-model ensemble ranking
    python run.py rank --ensemble --position positions/tt-research-2026.yaml

    # Viewer
    python run.py serve --port 8080
    python run.py serve --position positions/vap-teaching-2025.yaml  # Chat uses position context

    # Utilities
    python run.py status ../vap-search-2025/
    python run.py export ../vap-search-2025/ --output summary.csv
    python run.py cleanup-kbs                              # Delete all KBs from server
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_position_yaml
from src.models import Document, DocumentCategory, ApplicantProfile, Evaluation
from src.extraction import extract_all_documents
from src.analysis import categorize_all_documents, build_profile, evaluate_applicant
from src.llm import LLMClient


# ── Logging Setup ──
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Applicant Discovery ──
def discover_applicants(applicants_dir: Path) -> list[Path]:
    """
    Find applicant folders in the given directory.
    Each subfolder is treated as one applicant.
    If the path itself contains documents (no subfolders), treat it as a single applicant.
    """
    if not applicants_dir.is_dir():
        print(f"Error: {applicants_dir} is not a directory")
        sys.exit(1)

    # Check if this directory itself has documents (single applicant)
    has_docs = any(
        f.suffix.lower() in {".pdf", ".docx", ".doc", ".txt", ".rtf"}
        for f in applicants_dir.iterdir()
        if f.is_file()
    )
    has_subdirs = any(d.is_dir() for d in applicants_dir.iterdir() if not d.name.startswith("."))

    if has_docs and not has_subdirs:
        # This IS an applicant folder
        return [applicants_dir]

    # Otherwise, each subfolder is an applicant
    folders = sorted([
        d for d in applicants_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not folders:
        print(f"No applicant folders found in {applicants_dir}")
        sys.exit(1)

    return folders


def folder_to_name(folder: Path) -> str:
    """Convert folder name to display name (e.g., 'Alice_Smith' -> 'Alice Smith')."""
    return folder.name.replace("_", " ").replace("-", " ")


# ── Process Command ──
def process_applicants(args):
    """Process all applicants: extract, categorize, [RAG], profile, evaluate [ensemble]."""
    # Load config — from YAML position profile or defaults
    if hasattr(args, 'position') and args.position:
        config = load_position_yaml(args.position)
        print(f"📋 Position profile: {args.position}")
        print(f"   {config.position_summary()}")
        print(f"   Primary: {config.weights.primary}, Secondary: {config.weights.secondary}")
    else:
        config = Config()

    config.applicants_dir = Path(args.applicants_dir).resolve()
    if args.results_dir:
        config.results_dir = Path(args.results_dir).resolve()
    else:
        config.results_dir = PROJECT_ROOT / "data" / "results"

    config.results_dir.mkdir(parents=True, exist_ok=True)

    # CLI overrides for RAG and ensemble
    if args.no_rag:
        config.rag.enabled = False
    if args.no_ensemble:
        config.ensemble.enabled = False
    if args.force_recreate_kbs:
        config.rag.force_recreate = True

    # Initialize LLM client
    model = args.model or config.llm.extraction_model
    client = LLMClient(model=model)
    print(f"\n🔧 Using LLM model: {model}")

    # Initialize RAG manager if enabled
    rag_manager = None
    if config.rag.enabled:
        from src.rag import ApplicantKBManager
        rag_manager = ApplicantKBManager(
            client,
            index_wait=config.rag.index_wait_seconds,
            upload_settle=config.rag.upload_settle_seconds,
        )
        print(f"📚 RAG: Enabled (index wait: {config.rag.index_wait_seconds}s)")
    else:
        print(f"📚 RAG: Disabled (using text-stuffed prompts)")

    # Log ensemble status
    if config.ensemble.enabled:
        print(f"🤖 Ensemble: {len(config.ensemble.models)} models → {config.ensemble.synthesizer_model}")
        for m in config.ensemble.models:
            print(f"   • {m}")
    else:
        print(f"🤖 Ensemble: Disabled (single-model evaluation)")

    # Discover applicants
    folders = discover_applicants(config.applicants_dir)
    print(f"📂 Found {len(folders)} applicant(s) in {config.applicants_dir}\n")

    reprocess = getattr(args, 'reprocess', False)
    results = []
    skipped = 0

    for i, folder in enumerate(folders, 1):
        name = folder_to_name(folder)

        # Skip already-processed applicants unless --reprocess
        if not reprocess:
            eval_file = config.get_results_path(folder.name) / "evaluation.json"
            if eval_file.exists():
                skipped += 1
                print(f"  ⏭️  [{i}/{len(folders)}] Skipping (already processed): {name}")
                continue

        print(f"{'='*60}")
        print(f"[{i}/{len(folders)}] Processing: {name}")
        print(f"{'='*60}")

        try:
            result = _process_single_applicant(
                folder, name, client, config, rag_manager,
            )
            results.append(result)
            print(f"  ✅ Done: {result['recommendation']} "
                  f"(Teaching: {'⭐' * result['teaching_stars']}, "
                  f"Research: {'⭐' * result['research_stars']})\n")

        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to process {name}: {e}")
            print(f"  ❌ Error: {e}\n")
            results.append({
                "name": name,
                "folder": folder.name,
                "error": str(e),
            })

    if skipped:
        print(f"\n⏭️  Skipped {skipped} already-processed applicant(s). Use --reprocess to redo them.")

    # Print summary
    _print_summary(results)

    # Auto-run comparative ranking if we have results
    total_evaluated = len([r for r in results if "error" not in r]) + skipped
    if total_evaluated >= 2 and not getattr(args, 'skip_rank', False):
        print(f"\n{'='*60}")
        print("COMPARATIVE RANKING")
        print(f"{'='*60}")
        _run_ranking(config.results_dir, client, config)


def rank_applicants(args):
    """Run (or re-run) comparative ranking on already-processed results."""
    # Load config — from YAML position profile or defaults
    if hasattr(args, 'position') and args.position:
        config = load_position_yaml(args.position)
        print(f"📋 Position profile: {args.position}")
    else:
        config = Config()

    if args.results_dir:
        config.results_dir = Path(args.results_dir).resolve()
    else:
        config.results_dir = PROJECT_ROOT / "data" / "results"

    model = args.model or config.llm.extraction_model
    client = LLMClient(model=model)
    print(f"\n🔧 Using LLM model: {model}")

    # Ensemble for ranking
    ensemble_config = None
    if getattr(args, 'ensemble', False):
        ensemble_config = config.ensemble
        ensemble_config.enabled = True
        print(f"🔀 Ensemble ranking: {len(ensemble_config.models)} models → {ensemble_config.synthesizer_model}")

    _run_ranking(config.results_dir, client, config, ensemble_config=ensemble_config)


def _run_ranking(results_dir: Path, client: LLMClient, config: Config | None = None, ensemble_config=None):
    """Shared logic for comparative ranking."""
    from src.analysis.comparative import run_comparative_ranking

    result = run_comparative_ranking(results_dir, client, config=config, ensemble_config=ensemble_config)

    if not result:
        return

    primary = config.weights.primary if config else "teaching"

    print(f"\n📊 Comparative Rankings ({result['pool_size']} candidates)")
    print(f"   Pool avg teaching: {result['pool_stats']['avg_teaching_individual']:.1f}⭐  "
          f"research: {result['pool_stats']['avg_research_individual']:.1f}⭐")
    print(f"   Top tier: {result['pool_stats']['top_tier_count']}  "
          f"Strong: {result['pool_stats']['strong_tier_count']}")
    print()

    # Print ranked table
    tier_emoji = {"Top": "🏆", "Strong": "🟢", "Middle": "🟡", "Below Average": "🟠", "Weak": "🔴"}
    primary_rank_key = f"{primary}_rank"
    for r in result["rankings"]:
        emoji = tier_emoji.get(r["tier"], "⚪")
        print(f"  {emoji} #{r.get(primary_rank_key, 0):2d}  "
              f"T:{r['teaching_curved']}⭐(was {r['teaching_individual']}⭐)  "
              f"R:{r['research_curved']}⭐(was {r['research_individual']}⭐)  "
              f"{r['tier']:14s}  {r['name']}")

    print(f"\n  💾 Saved to {results_dir / 'comparative.json'}")


def _process_single_applicant(
    folder: Path,
    name: str,
    client: LLMClient,
    config: Config,
    rag_manager=None,
) -> dict:
    """
    Full processing pipeline for a single applicant.

    Pipeline:
      1. Extract documents (text from PDFs/DOCXs)
      2. Categorize (classify each document type — pre-RAG, uses text sample)
      3. Create RAG knowledge base (if enabled — upload + index)
      4. Build profile (extract structured data from CV + cover letter)
      5. Evaluate (teaching, research, fit — optionally via ensemble)
      6. Save results

    Returns a summary dict.
    """
    results_path = config.get_results_path(folder.name)
    kb_id = None

    # ── 1. Extract documents ──
    print(f"  📄 Extracting documents...")
    documents = extract_all_documents(folder)

    if not any(d.raw_text for d in documents):
        raise RuntimeError("No text could be extracted from any documents")

    # ── 2. Categorize (pre-RAG — uses filename hints + small text sample) ──
    print(f"  🏷️  Categorizing {len(documents)} documents...")
    documents = categorize_all_documents(documents, client)

    # ── 3. Create RAG knowledge base (if enabled) ──
    if rag_manager is not None and config.rag.enabled:
        print(f"  📚 Creating knowledge base...")
        try:
            kb_id = rag_manager.get_or_create_kb(
                applicant_name=name,
                folder_name=folder.name,
                source_folder=folder,
                results_path=results_path,
                force_recreate=config.rag.force_recreate,
            )
            print(f"  📚 KB ready: {kb_id[:12]}...")
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"RAG KB creation failed for {name}, falling back to classic mode: {e}"
            )
            kb_id = None

    # ── 4. Build profile ──
    print(f"  👤 Building profile...")
    profile = build_profile(
        applicant_name=name,
        folder_name=folder.name,
        documents=documents,
        client=client,
        position_config=config.position,
    )

    # ── 5. Evaluate (with optional RAG + ensemble) ──
    print(f"  ⭐ Evaluating...")
    ensemble_config = config.ensemble if config.ensemble.enabled else None
    evaluation = evaluate_applicant(
        profile,
        documents,
        client,
        kb_id=kb_id,
        ensemble_config=ensemble_config,
        config=config,
    )

    # ── 6. Save results ──
    _save_results(results_path, profile, evaluation, documents, folder, config)

    # Optionally clean up KB after processing
    if kb_id and not config.rag.keep_kbs_after_processing and rag_manager:
        print(f"  🧹 Cleaning up KB...")
        try:
            rag_manager.cleanup_applicant_kb(results_path)
        except Exception as e:
            logging.getLogger(__name__).warning(f"KB cleanup failed: {e}")

    return {
        "name": profile.name,
        "folder": folder.name,
        "teaching_stars": evaluation.teaching_stars,
        "research_stars": evaluation.research_stars,
        "fit_score": evaluation.fit_score,
        "recommendation": evaluation.overall_recommendation,
        "red_flags": len(evaluation.red_flags),
        "summary": evaluation.executive_summary,
        "rag_enabled": kb_id is not None,
        "ensemble_enabled": ensemble_config is not None,
    }


def _save_results(
    results_path: Path,
    profile: ApplicantProfile,
    evaluation: Evaluation,
    documents: list[Document],
    source_folder: Path,
    config: Config,
):
    """Save all results as JSON files."""
    # Profile
    with open(results_path / "profile.json", "w") as f:
        json.dump(profile.to_dict(), f, indent=2)

    # Evaluation (synthesized / final)
    with open(results_path / "evaluation.json", "w") as f:
        json.dump(evaluation.to_dict(), f, indent=2)

    # Ensemble details (per-model breakdowns, if available)
    ensemble_details = getattr(evaluation, "_ensemble_details", None)
    if ensemble_details and config.ensemble.save_individual_results:
        with open(results_path / "evaluation_ensemble.json", "w") as f:
            json.dump(
                {
                    "models": config.ensemble.models if config.ensemble.enabled else [],
                    "synthesizer_model": config.ensemble.synthesizer_model if config.ensemble.enabled else None,
                    "steps": ensemble_details,
                },
                f,
                indent=2,
                default=str,  # handle any non-serializable types
            )

    # Document metadata (without raw text — that stays in memory only)
    with open(results_path / "documents.json", "w") as f:
        json.dump(
            {
                "source_folder": str(source_folder),
                "documents": [d.to_dict() for d in documents],
            },
            f,
            indent=2,
        )

    # Processing metadata (what modes were used)
    with open(results_path / "processing_meta.json", "w") as f:
        from datetime import datetime
        json.dump(
            {
                "processed_at": datetime.now().isoformat(),
                "rag_enabled": config.rag.enabled,
                "ensemble_enabled": config.ensemble.enabled,
                "ensemble_models": config.ensemble.models if config.ensemble.enabled else [],
                "synthesizer_model": config.ensemble.synthesizer_model if config.ensemble.enabled else None,
                "evaluation_model": config.llm.evaluation_model,
                "position": {
                    "summary": config.position_summary(),
                    "title": config.position.position_title,
                    "focus": config.position.position_focus,
                    "duration": config.position.position_duration,
                    "department": config.position.department,
                    "university": config.position.university,
                    "primary_weight": config.weights.primary,
                    "secondary_weight": config.weights.secondary,
                },
            },
            f,
            indent=2,
        )

    print(f"  💾 Results saved to {results_path}")


def _print_summary(results: list[dict]):
    """Print a summary table of all processed applicants."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if successful:
        # Sort by teaching stars descending
        successful.sort(key=lambda r: (r["teaching_stars"], r["research_stars"]), reverse=True)

        for r in successful:
            flags = f" ⚠️{r['red_flags']} flags" if r["red_flags"] else ""
            mode = ""
            if r.get("rag_enabled"):
                mode += " 📚"
            if r.get("ensemble_enabled"):
                mode += " 🤖"
            print(
                f"  {'⭐' * r['teaching_stars']:10s} "
                f"T:{'⭐' * r['teaching_stars']} R:{'⭐' * r['research_stars']}  "
                f"{r['recommendation']:15s} {r['name']}{flags}{mode}"
            )

    if failed:
        print(f"\n  ❌ Failed ({len(failed)}):")
        for r in failed:
            print(f"     {r['name']}: {r['error']}")

    print(f"\nTotal: {len(successful)} processed, {len(failed)} failed")


# ── Cleanup KBs Command ──
def cleanup_kbs(args):
    """Delete all knowledge bases created for applicants."""
    print()
    print("⚠️  WARNING: This deletes all RAG knowledge bases permanently.")
    print("   For a safer experience with full confirmation prompts, use:")
    print("   python cleanup_rag.py")
    print()

    confirm = input("Continue anyway? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    config = Config()
    if args.results_dir:
        config.results_dir = Path(args.results_dir).resolve()
    else:
        config.results_dir = PROJECT_ROOT / "data" / "results"

    model = config.llm.extraction_model
    client = LLMClient(model=model)

    from src.rag import ApplicantKBManager
    manager = ApplicantKBManager(client)

    # Show status first
    statuses = manager.status_all(config.results_dir)
    has_kb = [s for s in statuses if s.get("has_kb")]

    if not has_kb:
        print("No knowledge bases found to clean up.")
        return

    print(f"Found {len(has_kb)} knowledge base(s):")
    for s in has_kb:
        print(f"  📚 {s['applicant_name']} (KB: {s['kb_id'][:12]}...)")

    confirm = input(f"\nType 'DELETE ALL KBS' to confirm: ")
    if confirm.strip() != 'DELETE ALL KBS':
        print("Cancelled.")
        return

    delete_files = input("Also delete uploaded files from server? (y/n) ")
    manager.cleanup_all(
        config.results_dir,
        delete_files=(delete_files.lower() == 'y'),
    )
    print(f"✅ Cleaned up {len(has_kb)} knowledge bases.")


# ── Serve Command ──
def serve_viewer(args):
    """Launch the web viewer with optional chat support."""
    results_dir = Path(args.results_dir).resolve() if args.results_dir else PROJECT_ROOT / "data" / "results"
    applicants_dir = Path(args.applicants_dir).resolve() if args.applicants_dir else None
    port = args.port or 5000

    print(f"🌐 Starting viewer on http://127.0.0.1:{port}")
    print(f"📂 Results: {results_dir}")
    if applicants_dir:
        print(f"📂 Applicants: {applicants_dir}")

    # Create LLM client for chat (unless --no-chat)
    llm_client = None
    if not getattr(args, 'no_chat', False):
        try:
            config = Config()
            model = args.model or config.llm.evaluation_model
            llm_client = LLMClient(model=model)
            print(f"💬 Chat: Enabled (model: {model})")
        except Exception as e:
            print(f"💬 Chat: Disabled (could not init LLM client: {e})")
    else:
        print(f"💬 Chat: Disabled (--no-chat)")

    # Load position config for chat context
    position_config = None
    if hasattr(args, 'position') and args.position:
        position_config = load_position_yaml(args.position)
        print(f"📋 Position profile: {args.position}")
        print(f"   {position_config.position_summary()}")

    from src.viewer.app import create_app
    app = create_app(
        results_dir=results_dir,
        applicants_dir=applicants_dir,
        llm_client=llm_client,
        position_config=position_config,
    )
    app.run(host="127.0.0.1", port=port, debug=True)


# ── Status Command ──
def show_status(args):
    """Show processing status for all applicants."""
    applicants_dir = Path(args.applicants_dir).resolve()
    results_dir = Path(args.results_dir).resolve() if args.results_dir else PROJECT_ROOT / "data" / "results"

    folders = discover_applicants(applicants_dir)

    print(f"\nStatus for {len(folders)} applicants:")
    print(f"{'Name':<30s} {'Documents':<12s} {'Processed':<12s} {'RAG':<8s} {'Ensemble':<10s}")
    print("-" * 72)

    for folder in folders:
        name = folder_to_name(folder)
        doc_count = sum(1 for f in folder.iterdir() if f.suffix.lower() in {".pdf", ".docx", ".doc", ".txt"})
        result_path = results_dir / folder.name
        processed = "✅" if (result_path / "profile.json").exists() else "❌"

        # Check RAG status
        rag_meta = result_path / "rag_metadata.json"
        rag_status = "📚" if rag_meta.exists() else "—"

        # Check ensemble status
        ensemble_file = result_path / "evaluation_ensemble.json"
        ensemble_status = "🤖" if ensemble_file.exists() else "—"

        print(f"  {name:<28s} {doc_count:<12d} {processed:<12s} {rag_status:<8s} {ensemble_status}")


# ── Export Command ──
def export_summary(args):
    """Export results to CSV."""
    applicants_dir = Path(args.applicants_dir).resolve()
    results_dir = Path(args.results_dir).resolve() if args.results_dir else PROJECT_ROOT / "data" / "results"
    output_file = args.output or "applicant_summary.csv"

    folders = discover_applicants(applicants_dir)

    rows = []
    for folder in folders:
        result_path = results_dir / folder.name
        profile_file = result_path / "profile.json"
        eval_file = result_path / "evaluation.json"

        if not profile_file.exists() or not eval_file.exists():
            continue

        with open(profile_file) as f:
            profile_raw = json.load(f)
        with open(eval_file) as f:
            evaluation = json.load(f)

        review = {}
        review_file = result_path / "human_review.json"
        if review_file.exists():
            with open(review_file) as f:
                review = json.load(f)

        # Merge profile with human overrides
        profile = dict(profile_raw)
        profile_overrides = review.get("profile_overrides", {})
        for field, value in profile_overrides.items():
            if value is not None and value != "":
                profile[field] = value

        # Load comparative data if available
        comp_file = result_path / "comparative.json"
        comp = {}
        if comp_file.exists():
            with open(comp_file) as f:
                comp = json.load(f)

        # Load processing metadata
        proc_meta = {}
        meta_file = result_path / "processing_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                proc_meta = json.load(f)

        rows.append({
            "Name": profile.get("name", ""),
            "Name (LLM)": profile_raw.get("name", ""),
            "Terminal Degree": profile.get("terminal_degree", ""),
            "Institution": profile.get("degree_institution", ""),
            "Degree Field": profile.get("degree_field", ""),
            "Degree Year": profile.get("degree_year", ""),
            "Degree Status": profile.get("degree_status", ""),
            "Current Position": profile.get("current_position", ""),
            "Current Institution": profile.get("current_institution", ""),
            "Statistics Adjacent": "Yes" if profile.get("is_statistics_adjacent") else "No",
            "Teaching Stars (LLM)": evaluation.get("teaching_stars", ""),
            "Research Stars (LLM)": evaluation.get("research_stars", ""),
            "Fit Score (LLM)": evaluation.get("fit_score", ""),
            "Recommendation (LLM)": evaluation.get("overall_recommendation", ""),
            "Teaching Curved": comp.get("teaching_curved", ""),
            "Research Curved": comp.get("research_curved", ""),
            "Teaching Rank": comp.get("teaching_rank", ""),
            "Research Rank": comp.get("research_rank", ""),
            "Tier": comp.get("tier", ""),
            "Comparative Notes": comp.get("comparative_notes", ""),
            "Teaching Stars (Human)": review.get("teaching_stars", ""),
            "Research Stars (Human)": review.get("research_stars", ""),
            "Fit Score (Human)": review.get("fit_score", ""),
            "Recommendation (Human)": review.get("overall_recommendation", ""),
            "Shortlisted": "Yes" if review.get("shortlist") else "No",
            "Reviewed": "Yes" if review.get("reviewed") else "No",
            "Committee Comments": review.get("comments", ""),
            "Teaching Notes": review.get("teaching_notes", ""),
            "Research Notes": review.get("research_notes", ""),
            "Red Flags": len(evaluation.get("red_flags", [])),
            "Courses as Instructor": sum(
                c.get("semesters", 1) for c in profile.get("courses_taught", [])
                if c.get("role") in ("instructor", "co-instructor")
            ),
            "AI in Education": "Yes" if profile.get("ai_in_education") else "No",
            "Technologies": ", ".join(profile.get("teaching_technologies", [])),
            "Research Areas": ", ".join(profile.get("research_areas", [])),
            "Executive Summary": evaluation.get("executive_summary", ""),
            "Has Profile Edits": "Yes" if profile_overrides else "No",
            "RAG Used": "Yes" if proc_meta.get("rag_enabled") else "No",
            "Ensemble Used": "Yes" if proc_meta.get("ensemble_enabled") else "No",
            "Ensemble Models": ", ".join(proc_meta.get("ensemble_models", [])),
        })

    if rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Exported {len(rows)} applicants to {output_file}")
    else:
        print("No processed results found to export.")


# ── Argument Parser ──
def main():
    parser = argparse.ArgumentParser(
        description="Job Application Summarizer for Faculty Searches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # process
    p_process = subparsers.add_parser("process", help="Process applicant documents")
    p_process.add_argument("applicants_dir", help="Path to applicants directory")
    p_process.add_argument("--position", help="Path to position YAML profile (e.g., positions/vap-teaching-2025.yaml)")
    p_process.add_argument("--model", help="LLM model name (GenAI Studio)")
    p_process.add_argument("--results-dir", help="Override results directory")
    p_process.add_argument("--reprocess", action="store_true",
                           help="Re-process applicants even if results exist")
    p_process.add_argument("--skip-rank", action="store_true",
                           help="Skip comparative ranking after processing")
    p_process.add_argument("--no-rag", action="store_true",
                           help="Disable RAG (use text-stuffed prompts instead)")
    p_process.add_argument("--no-ensemble", action="store_true",
                           help="Disable ensemble (use single-model evaluation)")
    p_process.add_argument("--force-recreate-kbs", action="store_true",
                           help="Delete and recreate knowledge bases even if they exist")

    # rank
    p_rank = subparsers.add_parser("rank", help="Run comparative ranking on processed results")
    p_rank.add_argument("--model", help="LLM model name (GenAI Studio)")
    p_rank.add_argument("--results-dir", help="Override results directory")
    p_rank.add_argument("--position", help="Path to position YAML profile")
    p_rank.add_argument("--ensemble", action="store_true",
                         help="Use multi-model ensemble for ranking (runs ranking across all ensemble models, then synthesizes)")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch web viewer")
    p_serve.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    p_serve.add_argument("--results-dir", help="Results directory")
    p_serve.add_argument("--applicants-dir", help="Applicants directory (for document links)")
    p_serve.add_argument("--model", help="LLM model for chat (default: evaluation model from config)")
    p_serve.add_argument("--no-chat", action="store_true",
                         help="Disable chat interface (viewer-only, no LLM needed)")
    p_serve.add_argument("--position", help="Path to position YAML profile (for chat context)")

    # status
    p_status = subparsers.add_parser("status", help="Show processing status")
    p_status.add_argument("applicants_dir", help="Path to applicants directory")
    p_status.add_argument("--results-dir", help="Override results directory")

    # export
    p_export = subparsers.add_parser("export", help="Export results to CSV")
    p_export.add_argument("applicants_dir", help="Path to applicants directory")
    p_export.add_argument("--results-dir", help="Override results directory")
    p_export.add_argument("--output", "-o", help="Output CSV filename")

    # cleanup-kbs
    p_cleanup = subparsers.add_parser("cleanup-kbs", help="Delete all knowledge bases from server")
    p_cleanup.add_argument("--results-dir", help="Override results directory")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "process":
        process_applicants(args)
    elif args.command == "rank":
        rank_applicants(args)
    elif args.command == "serve":
        serve_viewer(args)
    elif args.command == "status":
        show_status(args)
    elif args.command == "export":
        export_summary(args)
    elif args.command == "cleanup-kbs":
        cleanup_kbs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()