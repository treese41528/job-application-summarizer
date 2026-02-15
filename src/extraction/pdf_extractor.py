"""
PDF text extraction.

Uses pdfplumber as primary extractor (better layout/table handling),
with pypdf as fallback. Includes basic text cleaning.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str | Path) -> tuple[str, str]:
    """
    Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Tuple of (extracted_text, method_used).

    Raises:
        RuntimeError: If no PDF library is available or extraction fails.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Try pdfplumber first (better layout handling)
    text, method = _try_pdfplumber(file_path)
    if text:
        return text, method

    # Fallback to pypdf
    text, method = _try_pypdf(file_path)
    if text:
        return text, method

    # Fallback to pdftotext command line
    text, method = _try_pdftotext(file_path)
    if text:
        return text, method

    raise RuntimeError(
        f"Could not extract text from {file_path}. "
        "Install pdfplumber or pypdf: pip install pdfplumber pypdf"
    )


def _try_pdfplumber(file_path: Path) -> tuple[str, str]:
    """Try extraction with pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not available")
        return "", ""

    try:
        text_parts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.debug(f"Page {i+1} of {file_path.name}: no text extracted")

        text = "\n\n".join(text_parts)
        if text.strip():
            logger.info(f"Extracted {len(text)} chars from {file_path.name} via pdfplumber")
            return text, "pdfplumber"
        else:
            logger.warning(f"pdfplumber returned empty text for {file_path.name}")
            return "", ""
    except Exception as e:
        logger.warning(f"pdfplumber failed on {file_path.name}: {e}")
        return "", ""


def _try_pypdf(file_path: Path) -> tuple[str, str]:
    """Try extraction with pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.debug("pypdf not available")
        return "", ""

    try:
        reader = PdfReader(str(file_path))
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        text = "\n\n".join(text_parts)
        if text.strip():
            logger.info(f"Extracted {len(text)} chars from {file_path.name} via pypdf")
            return text, "pypdf"
        else:
            logger.warning(f"pypdf returned empty text for {file_path.name}")
            return "", ""
    except Exception as e:
        logger.warning(f"pypdf failed on {file_path.name}: {e}")
        return "", ""


def _try_pdftotext(file_path: Path) -> tuple[str, str]:
    """Try extraction with pdftotext command line tool (poppler-utils)."""
    import subprocess

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(file_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(
                f"Extracted {len(result.stdout)} chars from {file_path.name} via pdftotext"
            )
            return result.stdout, "pdftotext"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"pdftotext not available or timed out: {e}")

    return "", ""
