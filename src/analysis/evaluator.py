"""
Evaluator.

Generates star ratings for teaching and research, identifies red flags,
summarizes recommendation letters, and produces an overall assessment.

Supports two optional, combinable modes:

1. **RAG-grounded** (``kb_id`` provided):
   Uses shorter prompts that let the LLM retrieve full document text
   from the applicant's knowledge base via ``collections=[kb_id]``.

2. **Ensemble** (``ensemble_config`` provided and ``enabled``):
   Runs each evaluation step across multiple models independently,
   then a synthesizer model merges them into a single final assessment.

Both modes are **opt-in**.  When neither is supplied, behaviour is
identical to the original single-model, text-stuffed evaluator.
"""

import json
import logging
from typing import Optional

from ..models.document import Document, DocumentCategory
from ..models.applicant import ApplicantProfile
from ..models.evaluation import Evaluation, RedFlag, LetterSummary
from ..llm.client import LLMClient
from ..config import EnsembleConfig
from .prompts import (
    # Classic (text-stuffed)
    EVALUATE_TEACHING,
    EVALUATE_RESEARCH,
    EVALUATE_FIT_AND_FLAGS,
    SUMMARIZE_RECOMMENDATION_LETTER,
    # RAG-grounded
    RAG_EVALUATE_TEACHING,
    RAG_EVALUATE_RESEARCH,
    RAG_EVALUATE_FIT_AND_FLAGS,
    RAG_SUMMARIZE_RECOMMENDATION_LETTER,
    # Ensemble synthesis
    SYNTHESIZE_TEACHING,
    SYNTHESIZE_RESEARCH,
    SYNTHESIZE_FIT_AND_FLAGS,
    # Helper
    format_ensemble_results_for_synthesis,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════


def evaluate_applicant(
    profile: ApplicantProfile,
    documents: list[Document],
    client: LLMClient,
    kb_id: Optional[str] = None,
    ensemble_config: Optional[EnsembleConfig] = None,
) -> Evaluation:
    """
    Run the full evaluation pipeline for an applicant.

    Steps:
      1. Evaluate teaching  (primary criterion)
      2. Evaluate research  (secondary)
      3. Summarize recommendation letters
      4. Assess fit and identify red flags  (uses 1+2+3 results)

    Args:
        profile: Structured applicant profile.
        documents: Categorized documents with extracted text.
        client: LLM client.
        kb_id: Per-applicant knowledge base ID for RAG queries.
               If ``None``, falls back to text-stuffed prompts.
        ensemble_config: Ensemble settings.
               If ``None`` or ``enabled=False``, uses single-model eval.

    Returns:
        Complete ``Evaluation`` object.  When ensemble mode is used the
        object gains an ``_ensemble_details`` dict with per-step
        individual model outputs (for saving to evaluation_ensemble.json).
    """
    # ── Resolve active modes ──
    use_rag = kb_id is not None
    use_ensemble = ensemble_config is not None and ensemble_config.enabled

    mode_parts = []
    if use_rag:
        mode_parts.append(f"RAG (kb={kb_id[:8]}…)")
    if use_ensemble:
        model_count = len(ensemble_config.models)
        mode_parts.append(
            f"Ensemble ({model_count} models → {ensemble_config.synthesizer_model})"
        )
    if not mode_parts:
        mode_parts.append("Classic (single model, text-stuffed)")
    logger.info(f"  Evaluation mode: {', '.join(mode_parts)}")

    collections = [kb_id] if use_rag else None
    ens_cfg = ensemble_config if use_ensemble else None

    evaluation = Evaluation()
    ensemble_details: dict[str, list[dict]] = {}  # per-step individual outputs

    # ── Step 1: Teaching ──
    logger.info("  Evaluating teaching…")
    evaluation, individuals = _evaluate_teaching(
        evaluation, profile, documents, client, collections, ens_cfg,
    )
    if individuals:
        ensemble_details["teaching"] = individuals

    # ── Step 2: Research ──
    logger.info("  Evaluating research…")
    evaluation, individuals = _evaluate_research(
        evaluation, profile, documents, client, collections, ens_cfg,
    )
    if individuals:
        ensemble_details["research"] = individuals

    # ── Step 3: Recommendation Letters (single-model — no ensemble) ──
    letters = [
        d for d in documents
        if d.category == DocumentCategory.LETTER_OF_RECOMMENDATION
    ]
    if letters:
        logger.info(f"  Summarizing {len(letters)} recommendation letters…")
        evaluation = _summarize_letters(
            evaluation, profile, letters, client, collections,
        )
    else:
        logger.info("  No recommendation letters found")

    # ── Step 4: Fit Assessment & Red Flags ──
    logger.info("  Assessing fit and checking red flags…")
    evaluation, individuals = _evaluate_fit_and_flags(
        evaluation, profile, documents, client, collections, ens_cfg,
    )
    if individuals:
        ensemble_details["fit_and_flags"] = individuals

    if ensemble_details:
        evaluation._ensemble_details = ensemble_details

    return evaluation


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — TEACHING
# ════════════════════════════════════════════════════════════════════════════


def _evaluate_teaching(
    evaluation: Evaluation,
    profile: ApplicantProfile,
    documents: list[Document],
    client: LLMClient,
    collections: Optional[list[str]],
    ensemble_config: Optional[EnsembleConfig],
) -> tuple[Evaluation, list[dict]]:
    """Evaluate teaching.  Returns (evaluation, individual_model_results)."""

    prompt = _build_teaching_prompt(profile, documents, collections)

    if prompt is None:
        evaluation.teaching_stars = 1
        evaluation.teaching_weaknesses = [
            "No teaching information found in any documents"
        ]
        return evaluation, []

    try:
        if ensemble_config:
            data, individuals = _ensemble_evaluate(
                prompt=prompt,
                client=client,
                ensemble_config=ensemble_config,
                collections=collections,
                synthesis_template=SYNTHESIZE_TEACHING,
                synthesis_kwargs={"name": profile.name},
                key_fields=[
                    "teaching_stars", "teaching_strengths",
                    "teaching_weaknesses", "teaching_evidence",
                ],
            )
        else:
            data = client.query_json(prompt, temperature=0.2, collections=collections)
            individuals = []

        evaluation.teaching_stars = _clamp_stars(data.get("teaching_stars", 3))
        evaluation.teaching_strengths = data.get("teaching_strengths", [])
        evaluation.teaching_weaknesses = data.get("teaching_weaknesses", [])
        evaluation.teaching_evidence = data.get("teaching_evidence", [])

    except Exception as e:
        logger.error(f"  Teaching evaluation failed: {e}")
        evaluation.teaching_stars = 0
        evaluation.teaching_weaknesses = [f"Evaluation failed: {e}"]
        individuals = []

    return evaluation, individuals


def _build_teaching_prompt(
    profile: ApplicantProfile,
    documents: list[Document],
    collections: Optional[list[str]],
) -> Optional[str]:
    """Choose RAG or classic teaching prompt."""

    if collections:
        # ── RAG mode ──  Compute course counts safely from profile data
        courses = getattr(profile, "courses_taught", None) or []
        n_instructor = sum(
            1 for c in courses
            if isinstance(c, dict) and c.get("role", "").lower() in ("instructor", "co-instructor")
        )
        n_ta = sum(
            1 for c in courses
            if isinstance(c, dict) and c.get("role", "").lower() == "ta"
        )
        techs = getattr(profile, "teaching_technologies", None) or []
        ai_edu = getattr(profile, "ai_in_education", False)

        return RAG_EVALUATE_TEACHING.format(
            name=profile.name,
            terminal_degree=profile.terminal_degree,
            degree_status=profile.degree_status,
            degree_field=profile.degree_field,
            instructor_course_count=n_instructor,
            ta_course_count=n_ta,
            teaching_technologies=", ".join(techs) if techs else "None listed",
            ai_in_education="Yes" if ai_edu else "No",
        )

    # ── Classic mode ──
    cv_teaching, teaching_statement = _get_teaching_text(documents)
    if not cv_teaching and not teaching_statement:
        return None

    return EVALUATE_TEACHING.format(
        name=profile.name,
        terminal_degree=profile.terminal_degree,
        degree_status=profile.degree_status,
        degree_field=profile.degree_field,
        cv_teaching_text=cv_teaching or "(No CV teaching section found)",
        teaching_statement_text=teaching_statement or "(No teaching statement provided)",
    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — RESEARCH
# ════════════════════════════════════════════════════════════════════════════


def _evaluate_research(
    evaluation: Evaluation,
    profile: ApplicantProfile,
    documents: list[Document],
    client: LLMClient,
    collections: Optional[list[str]],
    ensemble_config: Optional[EnsembleConfig],
) -> tuple[Evaluation, list[dict]]:
    """Evaluate research.  Returns (evaluation, individual_model_results)."""

    prompt = _build_research_prompt(profile, documents, collections)

    try:
        if ensemble_config:
            data, individuals = _ensemble_evaluate(
                prompt=prompt,
                client=client,
                ensemble_config=ensemble_config,
                collections=collections,
                synthesis_template=SYNTHESIZE_RESEARCH,
                synthesis_kwargs={"name": profile.name},
                key_fields=[
                    "research_stars", "research_strengths", "research_weaknesses",
                ],
            )
        else:
            data = client.query_json(prompt, temperature=0.2, collections=collections)
            individuals = []

        evaluation.research_stars = _clamp_stars(data.get("research_stars", 3))
        evaluation.research_strengths = data.get("research_strengths", [])
        evaluation.research_weaknesses = data.get("research_weaknesses", [])

    except Exception as e:
        logger.error(f"  Research evaluation failed: {e}")
        evaluation.research_stars = 0
        evaluation.research_weaknesses = [f"Evaluation failed: {e}"]
        individuals = []

    return evaluation, individuals


def _build_research_prompt(
    profile: ApplicantProfile,
    documents: list[Document],
    collections: Optional[list[str]],
) -> str:
    """Choose RAG or classic research prompt."""

    if collections:
        return RAG_EVALUATE_RESEARCH.format(
            name=profile.name,
            degree_field=profile.degree_field,
            research_areas=", ".join(profile.research_areas) if profile.research_areas else "Not specified",
            publications_count=profile.publications_count,
        )

    research_text = ""
    for doc in documents:
        if doc.category == DocumentCategory.RESEARCH_STATEMENT and doc.raw_text:
            research_text = doc.raw_text[:4000]
            break

    return EVALUATE_RESEARCH.format(
        name=profile.name,
        degree_field=profile.degree_field,
        research_areas=", ".join(profile.research_areas) if profile.research_areas else "Not specified",
        publications_count=profile.publications_count,
        research_statement_text=research_text or "(No research statement provided)",
    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — RECOMMENDATION LETTERS  (single-model only — no ensemble)
# ════════════════════════════════════════════════════════════════════════════


def _summarize_letters(
    evaluation: Evaluation,
    profile: ApplicantProfile,
    letter_docs: list[Document],
    client: LLMClient,
    collections: Optional[list[str]],
) -> Evaluation:
    """Summarize each recommendation letter, then derive consensus."""

    for doc in letter_docs:
        if not doc.raw_text:
            continue

        try:
            if collections:
                prompt = RAG_SUMMARIZE_RECOMMENDATION_LETTER.format(
                    candidate_name=profile.name,
                    letter_filename=doc.filename,
                )
            else:
                prompt = SUMMARIZE_RECOMMENDATION_LETTER.format(
                    candidate_name=profile.name,
                    letter_text=doc.raw_text[:4000],
                )

            data = client.query_json(prompt, temperature=0.1, collections=collections)

            summary = LetterSummary(
                writer_name=data.get("writer_name"),
                writer_title=data.get("writer_title"),
                writer_institution=data.get("writer_institution"),
                relationship_to_candidate=data.get("relationship_to_candidate", ""),
                tone=data.get("tone", ""),
                key_points=data.get("key_points", []),
                teaching_comments=data.get("teaching_comments"),
                research_comments=data.get("research_comments"),
                concerns_raised=data.get("concerns_raised"),
                source_filename=doc.filename,
            )
            evaluation.letter_summaries.append(summary)

        except Exception as e:
            logger.warning(f"  Letter summary failed for {doc.filename}: {e}")

    # ── Derive consensus ──
    if evaluation.letter_summaries:
        tones = [ls.tone for ls in evaluation.letter_summaries if ls.tone]
        if not tones:
            evaluation.letter_consensus = "Letter tones could not be determined"
        elif all(t in ("Highly Positive", "Positive") for t in tones):
            evaluation.letter_consensus = (
                f"All {len(tones)} letters are positive about the candidate"
            )
        elif any(t in ("Negative", "Tepid") for t in tones):
            evaluation.letter_consensus = (
                f"Letters show mixed support ({', '.join(tones)})"
            )
        elif any(t == "Mixed" for t in tones):
            evaluation.letter_consensus = (
                f"Letters show some ambivalence ({', '.join(tones)})"
            )
        else:
            evaluation.letter_consensus = (
                f"Letters are generally supportive ({', '.join(tones)})"
            )

    return evaluation


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — FIT ASSESSMENT & RED FLAGS
# ════════════════════════════════════════════════════════════════════════════


def _evaluate_fit_and_flags(
    evaluation: Evaluation,
    profile: ApplicantProfile,
    documents: list[Document],
    client: LLMClient,
    collections: Optional[list[str]],
    ensemble_config: Optional[EnsembleConfig],
) -> tuple[Evaluation, list[dict]]:
    """Holistic fit assessment and red flag identification."""

    prompt = _build_fit_prompt(profile, documents, collections)

    try:
        if ensemble_config:
            data, individuals = _ensemble_evaluate(
                prompt=prompt,
                client=client,
                ensemble_config=ensemble_config,
                collections=collections,
                synthesis_template=SYNTHESIZE_FIT_AND_FLAGS,
                synthesis_kwargs={
                    "name": profile.name,
                    "teaching_stars": evaluation.teaching_stars,
                    "teaching_strengths": ", ".join(
                        evaluation.teaching_strengths[:3]
                    ) or "None identified",
                    "research_stars": evaluation.research_stars,
                    "letter_consensus": evaluation.letter_consensus or "No letters",
                },
                key_fields=[
                    "fit_score", "fit_rationale", "red_flags",
                    "overall_recommendation", "executive_summary",
                ],
            )
        else:
            data = client.query_json(prompt, temperature=0.2, collections=collections)
            individuals = []

        evaluation.fit_score = _clamp_stars(data.get("fit_score", 3))
        evaluation.fit_rationale = data.get("fit_rationale", "")

        for rf_data in data.get("red_flags", []):
            if isinstance(rf_data, dict):
                evaluation.red_flags.append(RedFlag(
                    category=rf_data.get("category", "Unknown"),
                    severity=rf_data.get("severity", "Minor"),
                    description=rf_data.get("description", ""),
                    source_document=rf_data.get("source_document", ""),
                ))

        evaluation.overall_recommendation = data.get(
            "overall_recommendation", "Moderate"
        )
        evaluation.executive_summary = data.get("executive_summary", "")

    except Exception as e:
        logger.error(f"  Fit evaluation failed: {e}")
        evaluation.overall_recommendation = "Error"
        evaluation.executive_summary = f"Evaluation error: {e}"
        individuals = []

    return evaluation, individuals


def _build_fit_prompt(
    profile: ApplicantProfile,
    documents: list[Document],
    collections: Optional[list[str]],
) -> str:
    """Choose RAG or classic fit/flags prompt."""

    # ── Build safe profile JSON (shared by both modes) ──
    profile_dict = profile.to_dict()
    profile_json_str = json.dumps(profile_dict, indent=2)
    if len(profile_json_str) > 3000:
        profile_dict["courses_taught"] = profile_dict.get("courses_taught", [])[:5]
        profile_dict["notable_publications"] = profile_dict.get("notable_publications", [])[:3]
        profile_dict["awards"] = profile_dict.get("awards", [])[:3]
        profile_dict["professional_service"] = profile_dict.get("professional_service", [])[:3]
        profile_json_str = json.dumps(profile_dict, indent=2)

    if collections:
        return RAG_EVALUATE_FIT_AND_FLAGS.format(profile_json=profile_json_str)

    # ── Classic: build document summaries ──
    doc_summaries = []
    for doc in documents:
        if doc.raw_text:
            preview = doc.raw_text[:500]
            doc_summaries.append(
                f"[{doc.category.display_name if doc.category else 'Unknown'} - "
                f"{doc.filename}]:\n{preview}"
            )

    return EVALUATE_FIT_AND_FLAGS.format(
        profile_json=profile_json_str,
        all_documents_summary="\n\n".join(doc_summaries)[:4000],
    )


# ════════════════════════════════════════════════════════════════════════════
# ENSEMBLE ENGINE
# ════════════════════════════════════════════════════════════════════════════


def _ensemble_evaluate(
    prompt: str,
    client: LLMClient,
    ensemble_config: EnsembleConfig,
    collections: Optional[list[str]],
    synthesis_template: str,
    synthesis_kwargs: dict,
    key_fields: list[str],
) -> tuple[dict, list[dict]]:
    """
    Core ensemble engine — used by each evaluation step.

    1. Run *prompt* on every model in ``ensemble_config.models``
    2. Format successful results into text
    3. Feed them to the synthesizer model via ``synthesis_template``
    4. Return ``(synthesized_data, individual_results)``

    If only one model succeeds, its output is returned directly (no
    synthesis step — there's nothing to merge).
    """
    models = ensemble_config.models
    logger.info(f"    Running ensemble across {len(models)} models…")

    # ── Step 1: Query each model ──
    individual_results = client.ensemble_query_json(
        prompt,
        models=models,
        temperature=ensemble_config.member_temperature,
        collections=collections,
    )

    successes = [r for r in individual_results if r.get("data") and not r.get("error")]
    logger.info(f"    Ensemble: {len(successes)}/{len(models)} succeeded")

    if not successes:
        errors = "; ".join(f"{r['model']}: {r['error']}" for r in individual_results)
        raise RuntimeError(f"All ensemble models failed. Errors: {errors}")

    # Short-circuit: only one success → use it directly
    if len(successes) == 1:
        logger.info("    Single model succeeded — skipping synthesis")
        return successes[0]["data"], individual_results

    # ── Step 2: Format for synthesis ──
    evaluations_text = format_ensemble_results_for_synthesis(
        successes, key_fields=key_fields,
    )

    # ── Step 3: Run synthesizer ──
    synthesis_prompt = synthesis_template.format(
        num_evaluations=len(successes),
        evaluations_text=evaluations_text,
        **synthesis_kwargs,
    )

    logger.info(f"    Synthesizing with {ensemble_config.synthesizer_model}…")
    synthesized = client.synthesize(
        synthesis_prompt,
        synthesizer_model=ensemble_config.synthesizer_model,
        temperature=ensemble_config.synthesizer_temperature,
        max_tokens=ensemble_config.synthesizer_max_tokens,
        collections=collections,
    )

    return synthesized, individual_results


# ════════════════════════════════════════════════════════════════════════════
# TEXT HELPERS  (classic mode only)
# ════════════════════════════════════════════════════════════════════════════


def _get_teaching_text(documents: list[Document]) -> tuple[str, str]:
    """
    Extract teaching-relevant text from documents.
    Returns ``(cv_teaching_text, teaching_statement_text)``.
    Only used in classic (non-RAG) mode.
    """
    cv_teaching = ""
    teaching_statement = ""

    for doc in documents:
        if doc.category == DocumentCategory.CV and doc.raw_text:
            text_lower = doc.raw_text.lower()
            for marker in ["teaching", "instruction", "courses taught"]:
                idx = text_lower.find(marker)
                if idx >= 0:
                    cv_teaching = doc.raw_text[max(0, idx - 50):idx + 3000]
                    break
            if not cv_teaching:
                cv_teaching = doc.raw_text[:2000]

        elif doc.category == DocumentCategory.TEACHING_STATEMENT and doc.raw_text:
            teaching_statement = doc.raw_text[:4000]

    return cv_teaching, teaching_statement


def _clamp_stars(value: int) -> int:
    """Ensure star rating is between 1 and 5."""
    try:
        return max(1, min(5, int(value)))
    except (ValueError, TypeError):
        return 3