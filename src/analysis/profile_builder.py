"""
Profile Builder.

Extracts a structured ApplicantProfile from the candidate's documents,
primarily the CV and cover letter. Uses LLM to parse unstructured text
into structured data.
"""

import json
import logging

from ..models.document import Document, DocumentCategory
from ..models.applicant import ApplicantProfile, CourseTaught
from ..llm.client import LLMClient
from ..extraction.text_cleaner import chunk_text
from .prompts import EXTRACT_PROFILE_FROM_CV, SUPPLEMENT_PROFILE_FROM_COVER_LETTER
from ..config import PositionConfig

logger = logging.getLogger(__name__)


def _get_document_by_category(
    documents: list[Document], category: DocumentCategory
) -> Document | None:
    """Find the first document matching a given category."""
    for doc in documents:
        if doc.category == category:
            return doc
    return None


def _get_documents_by_category(
    documents: list[Document], category: DocumentCategory
) -> list[Document]:
    """Find all documents matching a given category."""
    return [doc for doc in documents if doc.category == category]


def _check_statistics_adjacent(
    degree_field: str, position_config: PositionConfig
) -> bool:
    """
    Check if a degree field is statistics-adjacent.

    Does case-insensitive substring matching against the configured
    list of adjacent fields.
    """
    if not degree_field:
        return False

    field_lower = degree_field.lower().strip()
    for adj_field in position_config.statistics_adjacent_fields:
        if adj_field.lower() in field_lower or field_lower in adj_field.lower():
            return True
    return False


def build_profile(
    applicant_name: str,
    folder_name: str,
    documents: list[Document],
    client: LLMClient,
    position_config: PositionConfig | None = None,
) -> ApplicantProfile:
    """
    Build a structured applicant profile from their documents.

    Pipeline:
    1. Extract structured data from CV via LLM
    2. Cross-reference with cover letter for additional details
    3. Check degree adjacency
    4. Return populated ApplicantProfile

    Args:
        applicant_name: Display name (from folder name).
        folder_name: The actual folder name.
        documents: Categorized documents for this applicant.
        client: LLM client for extraction.
        position_config: Position criteria (defaults to PositionConfig()).

    Returns:
        Populated ApplicantProfile.
    """
    if position_config is None:
        position_config = PositionConfig()

    profile = ApplicantProfile(name=applicant_name, folder_name=folder_name)

    # ── Step 1: Extract from CV ──
    cv_doc = _get_document_by_category(documents, DocumentCategory.CV)
    if cv_doc and cv_doc.raw_text:
        logger.info(f"  Extracting profile from CV ({cv_doc.word_count} words)...")
        profile = _extract_from_cv(profile, cv_doc, client)
    else:
        logger.warning(f"  No CV found for {applicant_name}")

    # ── Step 2: Supplement from Cover Letter ──
    cl_doc = _get_document_by_category(documents, DocumentCategory.COVER_LETTER)
    if cl_doc and cl_doc.raw_text:
        logger.info(f"  Cross-referencing with cover letter...")
        profile = _supplement_from_cover_letter(profile, cl_doc, client)
    else:
        logger.info(f"  No cover letter found for {applicant_name}")

    # ── Step 3: Check statistics adjacency ──
    profile.is_statistics_adjacent = _check_statistics_adjacent(
        profile.degree_field, position_config
    )
    logger.info(
        f"  Degree field '{profile.degree_field}' -> "
        f"statistics-adjacent: {profile.is_statistics_adjacent}"
    )

    # ── Step 4: Use folder name if no name extracted ──
    if not profile.name or profile.name == "Unknown":
        profile.name = applicant_name

    return profile


def _extract_from_cv(
    profile: ApplicantProfile, cv_doc: Document, client: LLMClient
) -> ApplicantProfile:
    """
    Extract structured profile data from CV text using LLM.

    For long CVs, uses chunking to stay within context limits.
    """
    cv_text = cv_doc.raw_text

    # If CV is very long, chunk it - but try full text first
    chunks = chunk_text(cv_text, max_tokens=6000)

    if len(chunks) == 1:
        # Fits in one call
        prompt = EXTRACT_PROFILE_FROM_CV.format(cv_text=cv_text)
    else:
        # Use first chunk (usually has education + teaching) + last chunk (recent work)
        combined = chunks[0] + "\n\n[...middle sections omitted...]\n\n" + chunks[-1]
        prompt = EXTRACT_PROFILE_FROM_CV.format(cv_text=combined)
        logger.info(f"  CV chunked: using first + last of {len(chunks)} chunks")

    try:
        data = client.query_json(prompt, temperature=0.1)
        profile = _populate_profile_from_llm(profile, data)
    except Exception as e:
        logger.error(f"  CV extraction failed: {e}")

    return profile


def _populate_profile_from_llm(
    profile: ApplicantProfile, data: dict
) -> ApplicantProfile:
    """Populate an ApplicantProfile from LLM-extracted JSON data."""

    # Basic info
    profile.name = data.get("name") or profile.name
    profile.email = data.get("email")
    profile.current_position = data.get("current_position")
    profile.current_institution = data.get("current_institution")
    profile.website = data.get("website")

    # Education
    profile.terminal_degree = data.get("terminal_degree", "")
    profile.degree_field = data.get("degree_field", "")
    profile.degree_institution = data.get("degree_institution", "")
    profile.degree_year = data.get("degree_year")
    profile.degree_status = data.get("degree_status", "Unknown")
    profile.additional_degrees = data.get("additional_degrees", [])

    # Teaching
    courses_raw = data.get("courses_taught", [])
    profile.courses_taught = []
    for c in courses_raw:
        if isinstance(c, dict):
            profile.courses_taught.append(CourseTaught.from_dict(c))

    profile.teaching_experience_years = float(
        data.get("teaching_experience_years", 0)
    )
    profile.teaching_technologies = data.get("teaching_technologies", [])
    profile.ai_in_education = bool(data.get("ai_in_education", False))
    profile.ai_education_details = data.get("ai_education_details")

    # Research
    profile.research_areas = data.get("research_areas", [])
    profile.publications_count = int(data.get("publications_count", 0))
    profile.grants_count = int(data.get("grants_count", 0))
    profile.notable_publications = data.get("notable_publications", [])

    # Other
    profile.awards = data.get("awards", [])
    profile.professional_service = data.get("professional_service", [])
    profile.languages = data.get("languages", [])

    return profile


def _supplement_from_cover_letter(
    profile: ApplicantProfile, cl_doc: Document, client: LLMClient
) -> ApplicantProfile:
    """
    Cross-reference cover letter against existing profile to find
    additional details or corrections.
    """
    try:
        # Truncate profile JSON safely (keep valid JSON structure)
        profile_json_str = json.dumps(profile.to_dict(), indent=2)
        if len(profile_json_str) > 2000:
            # Build a minimal version with key fields only
            mini = {
                "name": profile.name,
                "terminal_degree": profile.terminal_degree,
                "degree_field": profile.degree_field,
                "degree_institution": profile.degree_institution,
                "degree_status": profile.degree_status,
                "current_position": profile.current_position,
                "teaching_technologies": profile.teaching_technologies,
                "ai_in_education": profile.ai_in_education,
                "research_areas": profile.research_areas[:3],
            }
            profile_json_str = json.dumps(mini, indent=2)

        prompt = SUPPLEMENT_PROFILE_FROM_COVER_LETTER.format(
            name=profile.name,
            existing_profile_json=profile_json_str,
            cover_letter_text=cl_doc.raw_text[:3000],
        )
        data = client.query_json(prompt, temperature=0.1)

        # Apply corrections (only to known profile fields)
        valid_fields = set(ApplicantProfile.__dataclass_fields__.keys())
        for correction in data.get("corrections", []):
            field_name = correction.get("field", "")
            new_value = correction.get("corrected_value")
            if field_name in valid_fields and new_value is not None:
                logger.info(
                    f"  Correction from cover letter: "
                    f"{field_name} = {new_value}"
                )
                setattr(profile, field_name, new_value)

        # Add new technologies
        new_techs = data.get("additional_teaching_technologies", [])
        for tech in new_techs:
            if tech and tech not in profile.teaching_technologies:
                profile.teaching_technologies.append(tech)

        # Update AI education info if cover letter has more detail
        if data.get("ai_in_education") and not profile.ai_in_education:
            profile.ai_in_education = True
        if data.get("ai_education_details") and not profile.ai_education_details:
            profile.ai_education_details = data["ai_education_details"]

    except Exception as e:
        logger.warning(f"  Cover letter cross-reference failed: {e}")

    return profile
