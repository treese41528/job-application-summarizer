"""
Document extraction package.

Discovers and extracts text from all supported document types
in an applicant's folder.
"""

import logging
from pathlib import Path

from ..models.document import Document
from .pdf_extractor import extract_text_from_pdf
from .docx_extractor import extract_text_from_docx
from .text_cleaner import clean_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".rtf"}


def discover_documents(folder: Path) -> list[Path]:
    """
    Find all supported documents in a folder (non-recursive).

    Args:
        folder: Path to an applicant's document folder.

    Returns:
        Sorted list of document paths.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    docs = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(f)
        elif f.is_file():
            logger.debug(f"Skipping unsupported file: {f.name}")

    logger.info(f"Found {len(docs)} documents in {folder.name}")
    return docs


def extract_document(file_path: Path) -> Document:
    """
    Extract text from a single document file.

    Routes to the appropriate extractor based on file extension,
    cleans the text, and returns a Document object.

    Args:
        file_path: Path to the document.

    Returns:
        Document object with extracted text.
    """
    ext = file_path.suffix.lower()
    errors = []
    raw_text = ""
    method = ""

    try:
        if ext == ".pdf":
            raw_text, method = extract_text_from_pdf(file_path)

        elif ext in (".docx", ".doc"):
            raw_text, method = extract_text_from_docx(file_path)

        elif ext == ".txt":
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            method = "direct_read"

        elif ext == ".rtf":
            # Try pandoc for RTF
            import subprocess
            result = subprocess.run(
                ["pandoc", str(file_path), "-t", "plain", "--wrap=none"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                raw_text = result.stdout
                method = "pandoc"
            else:
                errors.append(f"pandoc failed on RTF: {result.stderr}")

        else:
            errors.append(f"Unsupported extension: {ext}")

    except Exception as e:
        errors.append(str(e))
        logger.error(f"Extraction failed for {file_path.name}: {e}")

    # Clean the text
    cleaned_text = clean_text(raw_text) if raw_text else ""

    doc = Document(
        filename=file_path.name,
        filepath=file_path,
        file_extension=ext,
        raw_text=cleaned_text,
        word_count=len(cleaned_text.split()) if cleaned_text else 0,
        extraction_method=method,
        extraction_errors=errors,
    )

    if cleaned_text:
        logger.info(
            f"  {file_path.name}: {doc.word_count} words via {method}"
        )
    else:
        logger.warning(f"  {file_path.name}: NO TEXT EXTRACTED")

    return doc


def extract_all_documents(folder: Path) -> list[Document]:
    """
    Discover and extract text from all documents in a folder.

    Args:
        folder: Path to an applicant's document folder.

    Returns:
        List of Document objects.
    """
    paths = discover_documents(folder)
    documents = []
    for path in paths:
        doc = extract_document(path)
        documents.append(doc)

    extracted = sum(1 for d in documents if d.raw_text)
    logger.info(
        f"Extracted {extracted}/{len(documents)} documents from {folder.name}"
    )
    return documents
