"""Data models for the job application summarizer."""

from .document import Document, DocumentCategory
from .applicant import ApplicantProfile, CourseTaught
from .evaluation import Evaluation, RedFlag, LetterSummary

__all__ = [
    "Document",
    "DocumentCategory",
    "ApplicantProfile",
    "CourseTaught",
    "Evaluation",
    "RedFlag",
    "LetterSummary",
]
