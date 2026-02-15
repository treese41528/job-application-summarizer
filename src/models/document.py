"""
Document data model.

Represents an extracted document with its text content, category,
and metadata. Used throughout the pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class DocumentCategory(str, Enum):
    """Classification categories for job application documents."""
    CV = "cv"
    COVER_LETTER = "cover_letter"
    TEACHING_STATEMENT = "teaching_statement"
    RESEARCH_STATEMENT = "research_statement"
    LETTER_OF_RECOMMENDATION = "letter_of_recommendation"
    OTHER = "other"

    @property
    def display_name(self) -> str:
        """Human-readable name for display in the viewer."""
        names = {
            "cv": "CV / Curriculum Vitae",
            "cover_letter": "Cover Letter",
            "teaching_statement": "Teaching Statement",
            "research_statement": "Research Statement",
            "letter_of_recommendation": "Letter of Recommendation",
            "other": "Other Document",
        }
        return names.get(self.value, self.value)

    @property
    def short_name(self) -> str:
        names = {
            "cv": "CV",
            "cover_letter": "Cover Letter",
            "teaching_statement": "Teaching",
            "research_statement": "Research",
            "letter_of_recommendation": "Letter",
            "other": "Other",
        }
        return names.get(self.value, self.value)


@dataclass
class Document:
    """A single document extracted from an applicant's folder."""
    # File info
    filename: str
    filepath: Path
    file_extension: str

    # Extracted content
    raw_text: str
    word_count: int

    # Classification (populated by categorizer)
    category: Optional[DocumentCategory] = None
    category_confidence: float = 0.0
    category_reasoning: str = ""

    # Metadata
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_method: str = ""  # e.g., "pdfplumber", "pandoc", "python-docx"
    extraction_errors: list[str] = field(default_factory=list)

    @property
    def is_categorized(self) -> bool:
        return self.category is not None

    @property
    def text_preview(self) -> str:
        """First 200 characters of text for quick display."""
        if len(self.raw_text) <= 200:
            return self.raw_text
        return self.raw_text[:200] + "..."

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "filename": self.filename,
            "filepath": str(self.filepath),
            "file_extension": self.file_extension,
            "word_count": self.word_count,
            "category": self.category.value if self.category else None,
            "category_confidence": self.category_confidence,
            "category_reasoning": self.category_reasoning,
            "extracted_at": self.extracted_at,
            "extraction_method": self.extraction_method,
            "extraction_errors": self.extraction_errors,
        }

    @classmethod
    def from_dict(cls, data: dict, raw_text: str = "") -> "Document":
        """Deserialize from dictionary."""
        cat = DocumentCategory(data["category"]) if data.get("category") else None
        return cls(
            filename=data["filename"],
            filepath=Path(data["filepath"]),
            file_extension=data["file_extension"],
            raw_text=raw_text,
            word_count=data["word_count"],
            category=cat,
            category_confidence=data.get("category_confidence", 0.0),
            category_reasoning=data.get("category_reasoning", ""),
            extracted_at=data.get("extracted_at", ""),
            extraction_method=data.get("extraction_method", ""),
            extraction_errors=data.get("extraction_errors", []),
        )
