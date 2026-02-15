"""
Web Viewer - Flask Application.

Serves a modern dashboard for browsing applicant profiles, evaluations,
and original documents. Runs locally only.

Routes:
    /                           Dashboard (all applicants grid)
    /applicant/<folder_name>    Individual applicant detail
    /api/applicants             JSON list of all applicants
    /api/applicant/<name>       JSON detail for one applicant
    /documents/<name>/<file>    Serve original documents
"""

import json
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, abort, request

logger = logging.getLogger(__name__)


def create_app(
    results_dir: Path | str | None = None,
    applicants_dir: Path | str | None = None,
    llm_client=None,
    position_config=None,
) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        results_dir: Path to data/results/ with processed JSON.
        applicants_dir: Path to original applicant folders (for document serving).
        llm_client: Optional LLMClient instance for the chat interface.
                    If None, chat with applicant KBs is disabled.
        position_config: Optional Config with position details for chat context.
                         If None, tries to load from processing_meta.json.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    app.config["LLM_CLIENT"] = llm_client
    app.config["POSITION_CONFIG"] = position_config

    # Resolve paths
    if results_dir:
        app.config["RESULTS_DIR"] = Path(results_dir).resolve()
    else:
        app.config["RESULTS_DIR"] = Path(__file__).parent.parent.parent / "data" / "results"

    if applicants_dir:
        app.config["APPLICANTS_DIR"] = Path(applicants_dir).resolve()
    else:
        app.config["APPLICANTS_DIR"] = None

    # ── Chat System Prompt Builder ──

    def _build_chat_system_prompt(applicant_name: str, folder_name: str) -> str:
        """Build system prompt for RAG chat, using position config if available."""
        cfg = app.config.get("POSITION_CONFIG")

        if cfg:
            position_desc = cfg.position_summary()
            department = cfg.position.department
            university = cfg.position.university
        else:
            # Try loading from processing_meta.json
            meta_file = app.config["RESULTS_DIR"] / folder_name / "processing_meta.json"
            position_desc = "a faculty position"
            department = "the department"
            university = "the university"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    pos = meta.get("position", {})
                    position_desc = pos.get("summary", position_desc)
                    department = pos.get("department", department)
                    university = pos.get("university", university)
                except Exception:
                    pass

        return (
            f"You are a helpful assistant reviewing the job application of {applicant_name} "
            f"for {position_desc} in {department} at {university}. "
            f"You have access to all of this applicant's documents (CV, cover letter, "
            f"teaching statement, research statement, letters of recommendation) via a "
            f"knowledge base. Answer questions about this applicant based on their documents. "
            f"Be specific, cite document details, and note if information is not found."
        )

    # ── Helper Functions ──

    def _load_human_review(folder_name: str) -> dict:
        """Load the human review overlay for an applicant (or empty defaults)."""
        review_file = app.config["RESULTS_DIR"] / folder_name / "human_review.json"
        if review_file.exists():
            try:
                with open(review_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading human review for {folder_name}: {e}")
        return {}

    def _save_human_review(folder_name: str, review: dict) -> bool:
        """Save human review data for an applicant."""
        results_path = app.config["RESULTS_DIR"] / folder_name
        if not results_path.is_dir():
            return False
        try:
            with open(results_path / "human_review.json", "w") as f:
                json.dump(review, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving human review for {folder_name}: {e}")
            return False

    def _effective_profile(profile: dict, review: dict) -> dict:
        """
        Merge LLM-extracted profile with human overrides.
        Human values take precedence when present.
        """
        eff = dict(profile)
        overrides = review.get("profile_overrides", {})
        profile_overrides_applied = {}

        for field, value in overrides.items():
            if value is not None and value != "":
                profile_overrides_applied[field] = {"llm": profile.get(field), "human": value}
                eff[field] = value

        eff["_profile_overrides"] = profile_overrides_applied
        return eff

    def _effective_evaluation(evaluation: dict, review: dict) -> dict:
        """
        Merge LLM evaluation with human overrides.
        Human values take precedence when present and non-null.
        Returns a new dict with an 'overrides' sub-dict tracking what was changed.
        """
        eff = dict(evaluation)
        overrides = {}

        # Star ratings and recommendation
        for field in ("teaching_stars", "research_stars", "fit_score", "overall_recommendation"):
            if review.get(field) is not None:
                overrides[field] = {"llm": evaluation.get(field), "human": review[field]}
                eff[field] = review[field]

        # Eval list corrections (strengths, weaknesses, etc.)
        eval_corrections = review.get("eval_corrections", {})
        for field in ("teaching_strengths", "teaching_weaknesses", "teaching_evidence",
                      "research_strengths", "research_weaknesses", "executive_summary"):
            if field in eval_corrections:
                overrides[field] = {"llm": evaluation.get(field), "human": eval_corrections[field]}
                eff[field] = eval_corrections[field]

        eff["_overrides"] = overrides
        eff["_review"] = review
        return eff

    def _effective_profile(profile: dict, review: dict) -> dict:
        """
        Merge LLM profile with human corrections.
        Returns a new dict with corrections applied.
        """
        eff = dict(profile)
        corrections = review.get("profile_corrections", {})
        overrides = {}

        for field, value in corrections.items():
            if value is not None and value != "":
                overrides[field] = {"llm": profile.get(field), "human": value}
                eff[field] = value

        eff["_overrides"] = overrides
        return eff

    def _load_comparative_data() -> dict:
        """Load the pool-level comparative rankings if available."""
        comp_file = app.config["RESULTS_DIR"] / "comparative.json"
        if comp_file.exists():
            try:
                with open(comp_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading comparative data: {e}")
        return {}

    def _load_processing_meta(folder_name: str) -> dict:
        """Load processing metadata (RAG/ensemble info) for an applicant."""
        meta_file = app.config["RESULTS_DIR"] / folder_name / "processing_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading processing meta for {folder_name}: {e}")
        return {}

    def _load_ensemble_details(folder_name: str) -> dict:
        """Load per-model ensemble breakdowns for an applicant."""
        ensemble_file = app.config["RESULTS_DIR"] / folder_name / "evaluation_ensemble.json"
        if ensemble_file.exists():
            try:
                with open(ensemble_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading ensemble details for {folder_name}: {e}")
        return {}

    def _load_rag_metadata(folder_name: str) -> dict:
        """Load RAG knowledge base metadata for an applicant."""
        rag_file = app.config["RESULTS_DIR"] / folder_name / "rag_metadata.json"
        if rag_file.exists():
            try:
                with open(rag_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading RAG metadata for {folder_name}: {e}")
        return {}

    def _load_all_applicants() -> list[dict]:
        """Load summary data for all processed applicants."""
        results_path = app.config["RESULTS_DIR"]
        if not results_path.is_dir():
            return []

        # Load comparative rankings (indexed by folder_name)
        comp_data = _load_comparative_data()
        comp_map = {}
        for r in comp_data.get("rankings", []):
            comp_map[r.get("folder_name", "")] = r

        applicants = []
        for folder in sorted(results_path.iterdir()):
            if not folder.is_dir():
                continue

            profile_file = folder / "profile.json"
            eval_file = folder / "evaluation.json"
            docs_file = folder / "documents.json"

            if not profile_file.exists():
                continue

            try:
                with open(profile_file) as f:
                    profile = json.load(f)
                with open(eval_file) as f:
                    evaluation = json.load(f)

                review = _load_human_review(folder.name)
                proc_meta = _load_processing_meta(folder.name)

                documents = []
                if docs_file.exists():
                    with open(docs_file) as f:
                        docs_data = json.load(f)
                        documents = docs_data.get("documents", [])

                applicants.append({
                    "folder_name": folder.name,
                    "profile": _effective_profile(profile, review),
                    "profile_raw": profile,
                    "evaluation": _effective_evaluation(evaluation, review),
                    "documents": documents,
                    "review": review,
                    "comparative": comp_map.get(folder.name, {}),
                    "processing_meta": proc_meta,
                })
            except Exception as e:
                logger.warning(f"Error loading {folder.name}: {e}")

        return applicants

    def _load_single_applicant(folder_name: str) -> dict | None:
        """Load full data for a single applicant."""
        results_path = app.config["RESULTS_DIR"] / folder_name

        profile_file = results_path / "profile.json"
        eval_file = results_path / "evaluation.json"
        docs_file = results_path / "documents.json"

        if not profile_file.exists():
            return None

        try:
            with open(profile_file) as f:
                profile = json.load(f)
            with open(eval_file) as f:
                evaluation = json.load(f)

            review = _load_human_review(folder_name)
            proc_meta = _load_processing_meta(folder_name)
            ensemble_details = _load_ensemble_details(folder_name)
            rag_metadata = _load_rag_metadata(folder_name)

            documents = []
            source_folder = None
            if docs_file.exists():
                with open(docs_file) as f:
                    docs_data = json.load(f)
                    documents = docs_data.get("documents", [])
                    source_folder = docs_data.get("source_folder")

            # Load comparative data for this applicant
            comparative = {}
            comp_file = results_path / "comparative.json"
            if comp_file.exists():
                try:
                    with open(comp_file) as f:
                        comparative = json.load(f)
                except Exception:
                    pass

            return {
                "folder_name": folder_name,
                "profile": _effective_profile(profile, review),
                "profile_raw": profile,
                "evaluation": _effective_evaluation(evaluation, review),
                "evaluation_raw": evaluation,
                "review": review,
                "documents": documents,
                "source_folder": source_folder,
                "comparative": comparative,
                "processing_meta": proc_meta,
                "ensemble_details": ensemble_details,
                "rag_metadata": rag_metadata,
            }
        except Exception as e:
            logger.error(f"Error loading {folder_name}: {e}")
            return None

    def _resolve_applicant_folder(folder_name: str) -> Path | None:
        """Find the actual applicant folder for serving documents."""
        # First check the configured applicants directory
        if app.config["APPLICANTS_DIR"]:
            candidate = app.config["APPLICANTS_DIR"] / folder_name
            if candidate.is_dir():
                return candidate

        # Check if source_folder is stored in documents.json
        docs_file = app.config["RESULTS_DIR"] / folder_name / "documents.json"
        if docs_file.exists():
            try:
                with open(docs_file) as f:
                    data = json.load(f)
                source = data.get("source_folder")
                if source and Path(source).is_dir():
                    return Path(source)
            except Exception:
                pass

        return None

    # ── HTML Routes ──

    @app.route("/")
    def dashboard():
        """Main dashboard showing all applicants."""
        applicants = _load_all_applicants()

        # Parse sort/filter params
        sort_by = request.args.get("sort", "teaching")
        filter_rec = request.args.get("rec", "all")
        min_teaching = int(request.args.get("min_teaching", 0))
        filter_show = request.args.get("show", "all")

        # Apply filters
        if filter_rec != "all":
            applicants = [
                a for a in applicants
                if a["evaluation"].get("overall_recommendation", "").lower() == filter_rec.lower()
            ]

        if min_teaching > 0:
            applicants = [
                a for a in applicants
                if a["evaluation"].get("teaching_stars", 0) >= min_teaching
            ]

        if filter_show == "shortlisted":
            applicants = [a for a in applicants if a.get("review", {}).get("shortlist")]
        elif filter_show == "reviewed":
            applicants = [a for a in applicants if a.get("review", {}).get("reviewed")]
        elif filter_show == "unreviewed":
            applicants = [a for a in applicants if not a.get("review", {}).get("reviewed")]

        # Sort
        if sort_by == "teaching":
            applicants.sort(
                key=lambda a: (
                    a["evaluation"].get("teaching_stars", 0),
                    a["evaluation"].get("research_stars", 0),
                ),
                reverse=True,
            )
        elif sort_by == "research":
            applicants.sort(
                key=lambda a: a["evaluation"].get("research_stars", 0),
                reverse=True,
            )
        elif sort_by == "fit":
            applicants.sort(
                key=lambda a: a["evaluation"].get("fit_score", 0),
                reverse=True,
            )
        elif sort_by == "rank":
            tier_order = {"Top": 0, "Strong": 1, "Middle": 2, "Below Average": 3, "Weak": 4, "": 5, "Unranked": 5}
            applicants.sort(
                key=lambda a: (
                    tier_order.get(a.get("comparative", {}).get("tier", ""), 5),
                    a.get("comparative", {}).get("teaching_rank", 999),
                ),
            )
        elif sort_by == "name":
            applicants.sort(key=lambda a: a["profile"].get("name", ""))

        return render_template(
            "dashboard.html",
            applicants=applicants,
            total=len(applicants),
            sort_by=sort_by,
            filter_rec=filter_rec,
            min_teaching=min_teaching,
            filter_show=filter_show,
        )

    @app.route("/applicant/<folder_name>")
    def applicant_detail(folder_name: str):
        """Individual applicant detail page."""
        data = _load_single_applicant(folder_name)
        if not data:
            abort(404)

        # Check if we can serve documents
        doc_folder = _resolve_applicant_folder(folder_name)
        data["can_serve_documents"] = doc_folder is not None

        return render_template("applicant.html", data=data)

    # ── API Routes ──

    @app.route("/api/applicants")
    def api_applicants():
        """JSON endpoint: all applicants summary."""
        applicants = _load_all_applicants()
        return jsonify(applicants)

    @app.route("/api/applicant/<folder_name>")
    def api_applicant(folder_name: str):
        """JSON endpoint: single applicant detail."""
        data = _load_single_applicant(folder_name)
        if not data:
            abort(404)
        return jsonify(data)

    @app.route("/api/applicant/<folder_name>/review", methods=["POST"])
    def api_save_review(folder_name: str):
        """Save human review data for an applicant."""
        from datetime import datetime

        # Verify this applicant exists
        results_path = app.config["RESULTS_DIR"] / folder_name
        if not results_path.is_dir():
            abort(404)

        incoming = request.get_json(silent=True)
        if not incoming:
            return jsonify({"error": "Invalid JSON"}), 400

        # Load existing review to merge
        existing = _load_human_review(folder_name)

        # Evaluation overrides (with type coercion)
        eval_fields = {
            "teaching_stars": lambda v: max(1, min(5, int(v))) if v is not None else None,
            "research_stars": lambda v: max(1, min(5, int(v))) if v is not None else None,
            "fit_score": lambda v: max(1, min(5, int(v))) if v is not None else None,
            "overall_recommendation": lambda v: v if v in (
                "Strong", "Moderate", "Weak", "Do Not Advance", None
            ) else existing.get("overall_recommendation"),
        }

        # Comment fields
        comment_fields = {
            "comments": lambda v: str(v) if v else "",
            "teaching_notes": lambda v: str(v) if v else "",
            "research_notes": lambda v: str(v) if v else "",
        }

        # Profile corrections (string fields)
        profile_str_fields = {
            "name", "terminal_degree", "degree_field", "degree_institution",
            "degree_status", "current_position", "current_institution",
            "email", "website", "ai_education_details", "executive_summary",
        }

        # Profile corrections (int fields)
        profile_int_fields = {
            "degree_year": lambda v: int(v) if v else None,
            "publications_count": lambda v: int(v) if v else 0,
            "grants_count": lambda v: int(v) if v else 0,
        }

        # Profile corrections (bool fields)
        profile_bool_fields = {"is_statistics_adjacent", "ai_in_education"}

        # Profile corrections (list fields)
        profile_list_fields = {
            "research_areas", "teaching_technologies", "awards", "additional_degrees",
            "teaching_strengths", "teaching_weaknesses", "teaching_evidence",
            "research_strengths", "research_weaknesses",
        }

        # Initialize profile_overrides dict if needed
        if "profile_overrides" not in existing:
            existing["profile_overrides"] = {}
        if "eval_corrections" not in existing:
            existing["eval_corrections"] = {}

        # Process eval fields
        for field, coerce in eval_fields.items():
            if field in incoming:
                try:
                    existing[field] = coerce(incoming[field])
                except (ValueError, TypeError):
                    pass

        # Process comment fields
        for field, coerce in comment_fields.items():
            if field in incoming:
                try:
                    existing[field] = coerce(incoming[field])
                except (ValueError, TypeError):
                    pass

        # Process shortlist
        if "shortlist" in incoming:
            existing["shortlist"] = bool(incoming["shortlist"])

        # Process profile string corrections
        for field in profile_str_fields:
            if field in incoming:
                existing["profile_overrides"][field] = str(incoming[field]) if incoming[field] else ""

        # Process profile int corrections
        for field, coerce in profile_int_fields.items():
            if field in incoming:
                try:
                    existing["profile_overrides"][field] = coerce(incoming[field])
                except (ValueError, TypeError):
                    pass

        # Process profile bool corrections
        for field in profile_bool_fields:
            if field in incoming:
                existing["profile_overrides"][field] = bool(incoming[field])

        # Process profile list corrections
        for field in profile_list_fields:
            if field in incoming:
                val = incoming[field]
                if isinstance(val, list):
                    existing["profile_overrides"][field] = val
                elif isinstance(val, str):
                    # Split comma-separated string into list
                    existing["profile_overrides"][field] = [
                        x.strip() for x in val.split(",") if x.strip()
                    ]

        # Process eval list corrections (strengths/weaknesses stored in eval_corrections)
        eval_list_fields = {
            "teaching_strengths", "teaching_weaknesses", "teaching_evidence",
            "research_strengths", "research_weaknesses",
        }
        for field in eval_list_fields:
            if field in incoming:
                val = incoming[field]
                if isinstance(val, list):
                    existing["eval_corrections"][field] = val
                elif isinstance(val, str):
                    existing["eval_corrections"][field] = [
                        x.strip() for x in val.split("\n") if x.strip()
                    ]

        # Executive summary correction
        if "executive_summary" in incoming:
            existing["eval_corrections"]["executive_summary"] = str(incoming["executive_summary"])

        existing["reviewed"] = True
        existing["reviewed_at"] = datetime.now().isoformat()

        if _save_human_review(folder_name, existing):
            return jsonify({"ok": True, "review": existing})
        else:
            return jsonify({"error": "Save failed"}), 500

    @app.route("/api/applicant/<folder_name>/review/reset", methods=["POST"])
    def api_reset_review_field(folder_name: str):
        """Reset a specific human override back to LLM value."""
        results_path = app.config["RESULTS_DIR"] / folder_name
        if not results_path.is_dir():
            abort(404)

        incoming = request.get_json(silent=True) or {}
        field = incoming.get("field")

        existing = _load_human_review(folder_name)

        # Check which category this field belongs to
        eval_fields = {"teaching_stars", "research_stars", "fit_score", "overall_recommendation"}
        profile_fields = {
            "name", "terminal_degree", "degree_field", "degree_institution",
            "degree_status", "current_position", "current_institution",
            "email", "website", "ai_education_details", "degree_year",
            "publications_count", "grants_count", "is_statistics_adjacent",
            "ai_in_education", "research_areas", "teaching_technologies",
            "awards", "additional_degrees",
        }
        eval_list_fields = {
            "teaching_strengths", "teaching_weaknesses", "teaching_evidence",
            "research_strengths", "research_weaknesses", "executive_summary",
        }

        if field in eval_fields:
            existing.pop(field, None)
        elif field in profile_fields:
            if "profile_overrides" in existing:
                existing["profile_overrides"].pop(field, None)
        elif field in eval_list_fields:
            if "eval_corrections" in existing:
                existing["eval_corrections"].pop(field, None)
        else:
            return jsonify({"error": f"Unknown field: {field}"}), 400

        if _save_human_review(folder_name, existing):
            return jsonify({"ok": True, "review": existing})
        else:
            return jsonify({"error": "Save failed"}), 500

    # ── Chat with Applicant KB ──

    @app.route("/api/applicant/<folder_name>/chat", methods=["POST"])
    def api_chat_with_applicant(folder_name: str):
        """
        Chat with an applicant's RAG knowledge base.

        Expects JSON: {"message": "...", "history": [...]}
        Returns JSON: {"response": "...", "kb_id": "..."}
        """
        client = app.config.get("LLM_CLIENT")
        if not client:
            return jsonify({"error": "Chat not available — no LLM client configured. "
                            "Start with: python run.py serve --applicants-dir <dir>"}), 503

        # Load RAG metadata
        rag_meta = _load_rag_metadata(folder_name)
        kb_id = rag_meta.get("kb_id")

        if not kb_id:
            return jsonify({"error": f"No knowledge base found for this applicant. "
                            "Re-process with RAG enabled: python run.py process <dir>"}), 404

        incoming = request.get_json(silent=True) or {}
        message = incoming.get("message", "").strip()
        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Load profile for context
        profile_file = app.config["RESULTS_DIR"] / folder_name / "profile.json"
        applicant_name = folder_name.replace("_", " ")
        if profile_file.exists():
            try:
                with open(profile_file) as f:
                    profile = json.load(f)
                applicant_name = profile.get("name", applicant_name)
            except Exception:
                pass

        system_prompt = _build_chat_system_prompt(applicant_name, folder_name)

        try:
            response = client.query(
                prompt=message,
                system=system_prompt,
                collections=[kb_id],
                temperature=0.3,
            )
            return jsonify({
                "response": response,
                "kb_id": kb_id,
                "applicant_name": applicant_name,
            })
        except Exception as e:
            logger.error(f"Chat error for {folder_name}: {e}")
            return jsonify({"error": f"LLM query failed: {str(e)}"}), 500

    @app.route("/api/chat-status")
    def api_chat_status():
        """Check if chat functionality is available."""
        client = app.config.get("LLM_CLIENT")
        return jsonify({
            "chat_available": client is not None,
            "model": getattr(client, "model", None) if client else None,
        })

    # ── Document Serving ──

    @app.route("/documents/<folder_name>/<filename>")
    def serve_document(folder_name: str, filename: str):
        """Serve an original document file for viewing/download."""
        doc_folder = _resolve_applicant_folder(folder_name)
        if not doc_folder:
            abort(404)

        file_path = doc_folder / filename
        if not file_path.exists() or not file_path.is_file():
            abort(404)

        # Security: ensure file is within the applicant folder
        try:
            file_path.resolve().relative_to(doc_folder.resolve())
        except ValueError:
            abort(403)

        return send_from_directory(str(doc_folder), filename)

    # ── Template Filters ──

    @app.template_filter("stars")
    def stars_filter(value):
        """Convert int to star emoji string."""
        try:
            n = int(value)
            return "⭐" * n + "☆" * (5 - n)
        except (ValueError, TypeError):
            return "☆☆☆☆☆"

    @app.template_filter("rec_color")
    def rec_color_filter(value):
        """Map recommendation to CSS color class."""
        colors = {
            "Strong": "text-green-400",
            "Moderate": "text-yellow-400",
            "Weak": "text-orange-400",
            "Do Not Advance": "text-red-400",
        }
        return colors.get(value, "text-gray-400")

    @app.template_filter("rec_bg")
    def rec_bg_filter(value):
        """Map recommendation to background color class."""
        colors = {
            "Strong": "bg-green-900/30 border-green-700",
            "Moderate": "bg-yellow-900/30 border-yellow-700",
            "Weak": "bg-orange-900/30 border-orange-700",
            "Do Not Advance": "bg-red-900/30 border-red-700",
        }
        return colors.get(value, "bg-gray-900/30 border-gray-700")

    @app.template_filter("rec_emoji")
    def rec_emoji_filter(value):
        """Map recommendation to emoji."""
        emojis = {
            "Strong": "🟢",
            "Moderate": "🟡",
            "Weak": "🟠",
            "Do Not Advance": "🔴",
        }
        return emojis.get(value, "⚪")

    @app.template_filter("severity_color")
    def severity_color_filter(value):
        """Map red flag severity to color class."""
        colors = {
            "Minor": "text-yellow-300",
            "Moderate": "text-orange-300",
            "Serious": "text-red-300",
        }
        return colors.get(value, "text-gray-300")

    @app.template_filter("tone_color")
    def tone_color_filter(value):
        """Map letter tone to color class."""
        colors = {
            "Highly Positive": "text-green-300",
            "Positive": "text-green-400",
            "Mixed": "text-yellow-300",
            "Tepid": "text-orange-300",
            "Negative": "text-red-300",
        }
        return colors.get(value, "text-gray-300")

    @app.template_filter("tojson_pretty")
    def tojson_pretty_filter(value):
        """Pretty-print JSON for display in code blocks."""
        try:
            return json.dumps(value, indent=2, default=str)
        except (TypeError, ValueError):
            return str(value)

    return app