"""
Document Categorizer.

Uses an LLM to classify documents into categories:
CV, Cover Letter, Teaching Statement, Research Statement,
Letter of Recommendation, or Other.
"""

import logging
from ..models.document import Document, DocumentCategory
from ..llm.client import LLMClient
from .prompts import CATEGORIZE_DOCUMENT

logger = logging.getLogger(__name__)


# Filename-based hints for faster/cheaper classification
FILENAME_HINTS: dict[str, DocumentCategory] = {
    "cv": DocumentCategory.CV,
    "curriculum": DocumentCategory.CV,
    "vitae": DocumentCategory.CV,
    "resume": DocumentCategory.CV,
    "cover": DocumentCategory.COVER_LETTER,
    "letter_of_app": DocumentCategory.COVER_LETTER,
    "application_letter": DocumentCategory.COVER_LETTER,
    "teaching": DocumentCategory.TEACHING_STATEMENT,
    "teach_statement": DocumentCategory.TEACHING_STATEMENT,
    "research": DocumentCategory.RESEARCH_STATEMENT,
    "research_statement": DocumentCategory.RESEARCH_STATEMENT,
    "recommendation": DocumentCategory.LETTER_OF_RECOMMENDATION,
    "rec_letter": DocumentCategory.LETTER_OF_RECOMMENDATION,
    "letter_rec": DocumentCategory.LETTER_OF_RECOMMENDATION,
    "reference": DocumentCategory.LETTER_OF_RECOMMENDATION,
    "lor": DocumentCategory.LETTER_OF_RECOMMENDATION,
}


def categorize_by_filename(doc: Document) -> DocumentCategory | None:
    """
    Attempt to categorize a document based on its filename alone.
    Returns None if no confident match.
    """
    name_lower = doc.filename.lower().replace("-", "_").replace(" ", "_")
    stem = name_lower.rsplit(".", 1)[0]  # Remove extension

    for hint, category in FILENAME_HINTS.items():
        if hint in stem:
            logger.info(
                f"  {doc.filename} -> {category.value} (filename match: '{hint}')"
            )
            return category

    return None


def categorize_document(doc: Document, client: LLMClient) -> Document:
    """
    Categorize a single document using filename hints + LLM fallback.

    Modifies the document in-place and returns it.

    Args:
        doc: Document with extracted text.
        client: LLM client for classification.

    Returns:
        The document with category fields populated.
    """
    if not doc.raw_text:
        doc.category = DocumentCategory.OTHER
        doc.category_confidence = 0.0
        doc.category_reasoning = "No text could be extracted"
        return doc

    # Try filename-based classification first
    filename_category = categorize_by_filename(doc)
    if filename_category:
        doc.category = filename_category
        doc.category_confidence = 0.8
        doc.category_reasoning = f"Classified by filename pattern"
        # Still validate with LLM if we want to be sure
        # For now, trust filename hints to save API calls

    # Use LLM for documents without clear filename hints
    if doc.category is None:
        try:
            # Send first 3000 chars to save tokens
            text_sample = doc.raw_text[:3000]
            prompt = CATEGORIZE_DOCUMENT.format(document_text=text_sample)
            result = client.query_json(prompt, temperature=0.1)

            category_str = result.get("category", "other")
            try:
                doc.category = DocumentCategory(category_str)
            except ValueError:
                logger.warning(f"Unknown category '{category_str}', defaulting to OTHER")
                doc.category = DocumentCategory.OTHER

            doc.category_confidence = float(result.get("confidence", 0.0))
            doc.category_reasoning = result.get("reasoning", "")

            logger.info(
                f"  {doc.filename} -> {doc.category.value} "
                f"(LLM, confidence: {doc.category_confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"LLM categorization failed for {doc.filename}: {e}")
            doc.category = DocumentCategory.OTHER
            doc.category_confidence = 0.0
            doc.category_reasoning = f"LLM classification failed: {e}"

    return doc


def categorize_all_documents(
    documents: list[Document], client: LLMClient
) -> list[Document]:
    """
    Categorize all documents for an applicant.

    Args:
        documents: List of extracted documents.
        client: LLM client.

    Returns:
        The same list with categories populated.
    """
    logger.info(f"Categorizing {len(documents)} documents...")
    for doc in documents:
        categorize_document(doc, client)

    # Log summary
    categories = {}
    for doc in documents:
        cat = doc.category.value if doc.category else "unknown"
        categories[cat] = categories.get(cat, 0) + 1

    logger.info(f"  Categories: {categories}")
    return documents
