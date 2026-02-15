"""
Applicant profile data model.

Structured representation of an applicant extracted from their documents.
This is the core data model populated by the LLM profile builder.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CourseTaught:
    """A single course the applicant has taught or TA'd."""
    course_name: str
    course_number: Optional[str] = None
    level: str = "undergraduate"  # "undergraduate" or "graduate"
    role: str = "instructor"      # "instructor", "ta", "guest_lecturer", "co-instructor"
    institution: Optional[str] = None
    semesters: int = 1            # Number of times taught

    def to_dict(self) -> dict:
        return {
            "course_name": self.course_name,
            "course_number": self.course_number,
            "level": self.level,
            "role": self.role,
            "institution": self.institution,
            "semesters": self.semesters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CourseTaught":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApplicantProfile:
    """Structured profile of a job applicant, extracted via LLM from their documents."""

    # ── Basic Info ──
    name: str
    email: Optional[str] = None
    current_position: Optional[str] = None
    current_institution: Optional[str] = None
    website: Optional[str] = None

    # ── Education ──
    terminal_degree: str = ""                  # e.g., "Ph.D. in Statistics"
    degree_field: str = ""                     # e.g., "Statistics", "Biostatistics"
    degree_institution: str = ""
    degree_year: Optional[int] = None
    degree_status: str = "Unknown"             # "Completed", "Expected May 2025", "ABD"
    is_statistics_adjacent: bool = False
    additional_degrees: list[str] = field(default_factory=list)

    # ── Teaching ──
    courses_taught: list[CourseTaught] = field(default_factory=list)
    teaching_experience_years: float = 0.0
    teaching_technologies: list[str] = field(default_factory=list)  # "R", "Python", etc.
    ai_in_education: bool = False
    ai_education_details: Optional[str] = None

    # ── Research ──
    research_areas: list[str] = field(default_factory=list)
    publications_count: int = 0
    grants_count: int = 0
    notable_publications: list[str] = field(default_factory=list)

    # ── Other ──
    awards: list[str] = field(default_factory=list)
    professional_service: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    # ── Metadata ──
    folder_name: str = ""  # The directory name for this applicant

    @property
    def degree_display(self) -> str:
        """Formatted degree string for display."""
        parts = [self.terminal_degree]
        if self.degree_institution:
            parts.append(self.degree_institution)
        if self.degree_year:
            parts.append(str(self.degree_year))
        elif self.degree_status and self.degree_status != "Completed":
            parts.append(f"({self.degree_status})")
        return ", ".join(parts)

    @property
    def total_courses_as_instructor(self) -> int:
        return sum(
            c.semesters for c in self.courses_taught
            if c.role in ("instructor", "co-instructor")
        )

    @property
    def total_courses_as_ta(self) -> int:
        return sum(c.semesters for c in self.courses_taught if c.role == "ta")

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "name": self.name,
            "email": self.email,
            "current_position": self.current_position,
            "current_institution": self.current_institution,
            "website": self.website,
            "terminal_degree": self.terminal_degree,
            "degree_field": self.degree_field,
            "degree_institution": self.degree_institution,
            "degree_year": self.degree_year,
            "degree_status": self.degree_status,
            "is_statistics_adjacent": self.is_statistics_adjacent,
            "additional_degrees": self.additional_degrees,
            "courses_taught": [c.to_dict() for c in self.courses_taught],
            "teaching_experience_years": self.teaching_experience_years,
            "teaching_technologies": self.teaching_technologies,
            "ai_in_education": self.ai_in_education,
            "ai_education_details": self.ai_education_details,
            "research_areas": self.research_areas,
            "publications_count": self.publications_count,
            "grants_count": self.grants_count,
            "notable_publications": self.notable_publications,
            "awards": self.awards,
            "professional_service": self.professional_service,
            "languages": self.languages,
            "folder_name": self.folder_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ApplicantProfile":
        """Deserialize from dictionary."""
        courses = [CourseTaught.from_dict(c) for c in data.get("courses_taught", [])]
        return cls(
            name=data.get("name", "Unknown"),
            email=data.get("email"),
            current_position=data.get("current_position"),
            current_institution=data.get("current_institution"),
            website=data.get("website"),
            terminal_degree=data.get("terminal_degree", ""),
            degree_field=data.get("degree_field", ""),
            degree_institution=data.get("degree_institution", ""),
            degree_year=data.get("degree_year"),
            degree_status=data.get("degree_status", "Unknown"),
            is_statistics_adjacent=data.get("is_statistics_adjacent", False),
            additional_degrees=data.get("additional_degrees", []),
            courses_taught=courses,
            teaching_experience_years=data.get("teaching_experience_years", 0.0),
            teaching_technologies=data.get("teaching_technologies", []),
            ai_in_education=data.get("ai_in_education", False),
            ai_education_details=data.get("ai_education_details"),
            research_areas=data.get("research_areas", []),
            publications_count=data.get("publications_count", 0),
            grants_count=data.get("grants_count", 0),
            notable_publications=data.get("notable_publications", []),
            awards=data.get("awards", []),
            professional_service=data.get("professional_service", []),
            languages=data.get("languages", []),
            folder_name=data.get("folder_name", ""),
        )
