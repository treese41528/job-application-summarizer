"""
Evaluation data model.

Stores the LLM-generated assessment of an applicant including star ratings,
strengths/weaknesses, red flags, letter summaries, and overall recommendation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedFlag:
    """A potential concern identified in the application."""
    category: str          # e.g., "Degree Mismatch", "Employment Gap"
    severity: str          # "Minor", "Moderate", "Serious"
    description: str       # Detailed explanation
    source_document: str   # Which document this was found in

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "source_document": self.source_document,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RedFlag":
        return cls(**data)


@dataclass
class LetterSummary:
    """Summary of a single letter of recommendation."""
    writer_name: Optional[str] = None
    writer_title: Optional[str] = None
    writer_institution: Optional[str] = None
    relationship_to_candidate: str = ""  # "advisor", "teaching mentor", "colleague"
    tone: str = ""                        # "Highly Positive", "Positive", "Mixed", "Tepid"
    key_points: list[str] = field(default_factory=list)
    teaching_comments: Optional[str] = None
    research_comments: Optional[str] = None
    concerns_raised: Optional[str] = None
    source_filename: str = ""

    def to_dict(self) -> dict:
        return {
            "writer_name": self.writer_name,
            "writer_title": self.writer_title,
            "writer_institution": self.writer_institution,
            "relationship_to_candidate": self.relationship_to_candidate,
            "tone": self.tone,
            "key_points": self.key_points,
            "teaching_comments": self.teaching_comments,
            "research_comments": self.research_comments,
            "concerns_raised": self.concerns_raised,
            "source_filename": self.source_filename,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LetterSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Evaluation:
    """Complete evaluation of an applicant."""

    # ── Teaching Rating (Primary) ──
    teaching_stars: int = 0                          # 1-5
    teaching_strengths: list[str] = field(default_factory=list)
    teaching_weaknesses: list[str] = field(default_factory=list)
    teaching_evidence: list[str] = field(default_factory=list)  # Specific supporting facts

    # ── Research Rating (Secondary) ──
    research_stars: int = 0                          # 1-5
    research_strengths: list[str] = field(default_factory=list)
    research_weaknesses: list[str] = field(default_factory=list)

    # ── Fit Assessment ──
    fit_score: int = 0                               # 1-5
    fit_rationale: str = ""

    # ── Red Flags ──
    red_flags: list[RedFlag] = field(default_factory=list)

    # ── Letters of Recommendation ──
    letter_summaries: list[LetterSummary] = field(default_factory=list)
    letter_consensus: str = ""  # Overall consensus across all letters

    # ── Overall Assessment ──
    overall_recommendation: str = ""  # "Strong", "Moderate", "Weak", "Do Not Advance"
    executive_summary: str = ""       # 2-3 sentence summary

    @property
    def has_red_flags(self) -> bool:
        return len(self.red_flags) > 0

    @property
    def serious_red_flags(self) -> list[RedFlag]:
        return [rf for rf in self.red_flags if rf.severity == "Serious"]

    @property
    def recommendation_color(self) -> str:
        """Color code for the viewer UI."""
        colors = {
            "Strong": "green",
            "Moderate": "yellow",
            "Weak": "orange",
            "Do Not Advance": "red",
        }
        return colors.get(self.overall_recommendation, "gray")

    @property
    def recommendation_emoji(self) -> str:
        emojis = {
            "Strong": "🟢",
            "Moderate": "🟡",
            "Weak": "🟠",
            "Do Not Advance": "🔴",
        }
        return emojis.get(self.overall_recommendation, "⚪")

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "teaching_stars": self.teaching_stars,
            "teaching_strengths": self.teaching_strengths,
            "teaching_weaknesses": self.teaching_weaknesses,
            "teaching_evidence": self.teaching_evidence,
            "research_stars": self.research_stars,
            "research_strengths": self.research_strengths,
            "research_weaknesses": self.research_weaknesses,
            "fit_score": self.fit_score,
            "fit_rationale": self.fit_rationale,
            "red_flags": [rf.to_dict() for rf in self.red_flags],
            "letter_summaries": [ls.to_dict() for ls in self.letter_summaries],
            "letter_consensus": self.letter_consensus,
            "overall_recommendation": self.overall_recommendation,
            "executive_summary": self.executive_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evaluation":
        """Deserialize from dictionary."""
        red_flags = [RedFlag.from_dict(rf) for rf in data.get("red_flags", [])]
        letters = [LetterSummary.from_dict(ls) for ls in data.get("letter_summaries", [])]
        return cls(
            teaching_stars=data.get("teaching_stars", 0),
            teaching_strengths=data.get("teaching_strengths", []),
            teaching_weaknesses=data.get("teaching_weaknesses", []),
            teaching_evidence=data.get("teaching_evidence", []),
            research_stars=data.get("research_stars", 0),
            research_strengths=data.get("research_strengths", []),
            research_weaknesses=data.get("research_weaknesses", []),
            fit_score=data.get("fit_score", 0),
            fit_rationale=data.get("fit_rationale", ""),
            red_flags=red_flags,
            letter_summaries=letters,
            letter_consensus=data.get("letter_consensus", ""),
            overall_recommendation=data.get("overall_recommendation", ""),
            executive_summary=data.get("executive_summary", ""),
        )
