#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                     RAG KNOWLEDGE BASE CLEANUP                      ║
║                                                                      ║
║  This script permanently deletes ALL knowledge bases and uploaded    ║
║  files from the GenAI Studio server that were created during the     ║
║  applicant review process.                                           ║
║                                                                      ║
║  ⚠️  ONLY RUN THIS WHEN THE HIRING PROCESS IS FULLY COMPLETE ⚠️     ║
║                                                                      ║
║  After cleanup:                                                      ║
║    - The chat interface will no longer work for any applicant        ║
║    - RAG-grounded re-evaluation will require full reprocessing       ║
║    - Uploaded files will be removed from the GenAI Studio server     ║
║    - Local results (profiles, evaluations, reviews) are NOT deleted  ║
║                                                                      ║
║  This action CANNOT be undone.                                       ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python cleanup_rag.py                        # Interactive (default results dir)
    python cleanup_rag.py --results-dir <path>   # Custom results directory
    python cleanup_rag.py --dry-run              # Preview only, delete nothing
    python cleanup_rag.py --yes                  # Skip confirmations (DANGEROUS)
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI colors
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"  # No Color


def main():
    parser = argparse.ArgumentParser(
        description="Permanently delete all RAG knowledge bases from GenAI Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        help="Path to results directory (default: data/results/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip all confirmation prompts (use with extreme caution)",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Delete KBs but keep uploaded files on the server",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve() if args.results_dir else PROJECT_ROOT / "data" / "results"

    if not results_dir.is_dir():
        print(f"{RED}Error: Results directory not found: {results_dir}{NC}")
        sys.exit(1)

    # ── Banner ──
    print()
    print(f"{RED}{'='*66}")
    print(f"  ⚠️   RAG KNOWLEDGE BASE CLEANUP — PERMANENT DELETION   ⚠️")
    print(f"{'='*66}{NC}")
    print()

    if args.dry_run:
        print(f"{CYAN}  🔍 DRY RUN MODE — nothing will be deleted{NC}")
        print()

    # ── Scan for KBs ──
    print(f"{DIM}Scanning {results_dir} for knowledge bases...{NC}")
    print()

    from src.config import Config
    from src.llm import LLMClient
    from src.rag import ApplicantKBManager
    from src.rag.manager import RAGMetadata

    # Gather metadata without connecting to server yet
    applicants_with_kb = []
    applicants_without_kb = []
    total_files = 0

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        meta = RAGMetadata.load(folder)
        if meta and meta.kb_id:
            file_count = len(meta.file_ids)
            total_files += file_count
            applicants_with_kb.append({
                "folder": folder.name,
                "name": meta.applicant_name,
                "kb_id": meta.kb_id,
                "kb_name": meta.kb_name,
                "file_count": file_count,
                "created_at": meta.created_at,
            })
        else:
            applicants_without_kb.append(folder.name)

    if not applicants_with_kb:
        print(f"{GREEN}  ✅ No knowledge bases found. Nothing to clean up.{NC}")
        print()
        sys.exit(0)

    # ── Display summary ──
    print(f"  Found {BOLD}{len(applicants_with_kb)}{NC} knowledge base(s):")
    print()

    for a in applicants_with_kb:
        print(f"    📚 {BOLD}{a['name']}{NC}")
        print(f"       KB: {DIM}{a['kb_id']}{NC}")
        print(f"       Files: {a['file_count']}   Created: {a['created_at'][:10] if a['created_at'] else 'unknown'}")
        print()

    print(f"  {BOLD}Total:{NC} {len(applicants_with_kb)} KBs, {total_files} uploaded files")
    if applicants_without_kb:
        print(f"  {DIM}({len(applicants_without_kb)} applicants have no KB — unaffected){NC}")
    print()

    # ── What will happen ──
    print(f"{YELLOW}  This will permanently:{NC}")
    print(f"    ✕ Delete {len(applicants_with_kb)} knowledge base(s) from GenAI Studio")
    if not args.keep_files:
        print(f"    ✕ Delete {total_files} uploaded file(s) from GenAI Studio")
    else:
        print(f"    • Keep uploaded files on server (--keep-files)")
    print(f"    ✕ Remove local rag_metadata.json files")
    print()
    print(f"{GREEN}  This will NOT affect:{NC}")
    print(f"    ✓ Local evaluation results (profile.json, evaluation.json)")
    print(f"    ✓ Human reviews (human_review.json)")
    print(f"    ✓ Comparative rankings (comparative.json)")
    print(f"    ✓ Original applicant documents")
    print()

    if args.dry_run:
        print(f"{CYAN}  🔍 DRY RUN — exiting without changes.{NC}")
        print()
        sys.exit(0)

    # ── Confirmation gauntlet ──
    if not args.yes:
        # Confirmation 1
        print(f"{RED}{'─'*66}{NC}")
        print(f"{RED}  WARNING: This action CANNOT be undone.{NC}")
        print(f"{RED}  The chat interface will stop working for all applicants.{NC}")
        print(f"{RED}  RAG-grounded re-evaluation will require full reprocessing.{NC}")
        print(f"{RED}{'─'*66}{NC}")
        print()

        response = input(f"  {BOLD}Is the hiring process fully complete? (yes/no): {NC}").strip().lower()
        if response != "yes":
            print(f"\n  {GREEN}Cancelled. No changes made.{NC}")
            print(f"  {DIM}Run with --dry-run to preview without deleting.{NC}\n")
            sys.exit(0)

        # Confirmation 2
        print()
        response = input(
            f"  {BOLD}Type 'DELETE ALL KBS' to confirm permanent deletion: {NC}"
        ).strip()
        if response != "DELETE ALL KBS":
            print(f"\n  {GREEN}Cancelled. No changes made.{NC}\n")
            sys.exit(0)

        print()
        print(f"  {DIM}Starting in 3 seconds... (Ctrl+C to abort){NC}")
        try:
            for i in range(3, 0, -1):
                print(f"  {DIM}  {i}...{NC}")
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n\n  {GREEN}Aborted. No changes made.{NC}\n")
            sys.exit(0)

    # ── Execute cleanup ──
    print()
    print(f"  {YELLOW}Connecting to GenAI Studio...{NC}")

    config = Config()
    client = LLMClient(model=config.llm.extraction_model)
    manager = ApplicantKBManager(client)

    delete_files = not args.keep_files
    succeeded = 0
    failed = 0

    for a in applicants_with_kb:
        folder_path = results_dir / a["folder"]
        print(f"\n  🗑️  Cleaning up: {a['name']}")
        print(f"     KB: {a['kb_id'][:16]}...")

        try:
            success = manager.cleanup_applicant_kb(
                folder_path,
                delete_files=delete_files,
            )
            if success:
                succeeded += 1
                print(f"     {GREEN}✅ Deleted{NC}")
            else:
                failed += 1
                print(f"     {YELLOW}⚠️  Partial failure (some resources may remain){NC}")
        except Exception as e:
            failed += 1
            print(f"     {RED}❌ Error: {e}{NC}")

    # ── Final report ──
    print()
    print(f"{'='*66}")
    print(f"  CLEANUP COMPLETE")
    print(f"{'='*66}")
    print()
    print(f"  {GREEN}✅ Successfully cleaned: {succeeded}{NC}")
    if failed:
        print(f"  {RED}❌ Failed: {failed}{NC}")
        print(f"  {DIM}Check GenAI Studio manually for orphaned resources.{NC}")
    print()

    if succeeded == len(applicants_with_kb):
        print(f"  {GREEN}All knowledge bases have been removed.{NC}")
        print(f"  {DIM}Local results are untouched. The viewer will still work{NC}")
        print(f"  {DIM}for browsing profiles and evaluations, but the chat{NC}")
        print(f"  {DIM}interface is no longer available.{NC}")
    print()


if __name__ == "__main__":
    main()