"""
Configuration for Job Application Summarizer.

Central configuration for LLM models, position criteria, rating rubrics,
RAG settings, ensemble evaluation, and application settings.

There are two ways to customize for a specific search:

1. **Edit this file directly** — change defaults in the dataclasses below.
2. **Use a position YAML profile** — pass ``--position positions/my-search.yaml``
   on the CLI.  The YAML overrides position, rubric, red-flag, adjacent-field,
   and weighting settings without touching this file.

See ``positions/`` for example YAML profiles.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM model and parameter settings."""
    # Which models to use for each task (Purdue GenAI Studio model names)
    classification_model: str = "llama3.1:70b-instruct-q4_K_M"
    extraction_model: str = "llama3.1:70b-instruct-q4_K_M"
    evaluation_model: str = "llama3.1:70b-instruct-q4_K_M"

    # Generation parameters
    extraction_temperature: float = 0.1   # Low for consistent structured output
    evaluation_temperature: float = 0.3   # Slightly higher for nuanced assessment
    max_tokens: int = 4096
    max_tokens_per_chunk: int = 4000      # For chunking long documents


@dataclass
class EnsembleConfig:
    """
    Multi-model ensemble evaluation settings.

    When enabled, each evaluation step (teaching, research, fit) is run
    independently by every model in ``models``, then a synthesizer model
    merges the results into a single final assessment.

    Benefits:
    - Reduced single-model bias
    - More robust star ratings (evidence-weighted, not simply averaged)
    - Richer strengths/weaknesses (union of unique findings across models)
    - Per-model breakdowns saved for transparency and auditability
    """
    enabled: bool = False

    # Models that each independently evaluate the candidate.
    models: list[str] = field(default_factory=lambda: [
        "llama3.3:70b",       # Meta dense 70B
        "llama4:latest",      # Meta MoE — different architecture
        "qwen2.5:72b",        # Alibaba
        "gemma3:27b",         # Google
    ])

    # Model that reads all individual outputs and writes the final evaluation.
    # deepseek-r1 is well-suited for synthesis thanks to chain-of-thought.
    synthesizer_model: str = "deepseek-r1:70b"

    # Temperature for ensemble member queries (low for consistency)
    member_temperature: float = 0.2

    # Temperature for the synthesizer (low — we want reliable merging)
    synthesizer_temperature: float = 0.1

    # Max tokens for synthesizer output
    synthesizer_max_tokens: int = 4096

    # If True, save per-model outputs in evaluation_ensemble.json
    save_individual_results: bool = True


@dataclass
class RAGConfig:
    """
    Retrieval-Augmented Generation settings.

    When enabled, each applicant's documents are uploaded to GenAI Studio
    and indexed into a per-applicant knowledge base.  Evaluation prompts
    then reference the KB via ``collections=[kb_id]`` instead of stuffing
    truncated text into the prompt.

    Benefits over classic text-stuffed prompts:
    - Full document access (no 3000–4000 char truncation)
    - Better evidence citation (LLM retrieves specific passages)
    - Enables a post-processing chat interface over applicant docs
    """
    enabled: bool = True

    # Seconds to wait after linking files to KB for server-side indexing.
    # Too short → retrieval misses documents that haven't been embedded yet.
    index_wait_seconds: int = 12

    # Seconds to wait between upload and linking (lets upload finalize)
    upload_settle_seconds: int = 2

    # Keep KBs after processing?  Must be True for the chat interface.
    keep_kbs_after_processing: bool = True

    # If True, delete and recreate KBs even when saved metadata says one exists.
    force_recreate: bool = False


@dataclass
class PositionConfig:
    """Details about the position being hired for."""
    position_title: str = "Visiting Assistant Professor"
    position_focus: str = "Teaching (Primary)"
    position_duration: str = "2-year term"
    department: str = "Department of Statistics"
    university: str = "Purdue University"
    description: str = (
        "A temporary 2-year Visiting Assistant Professor position with a "
        "primary focus on teaching undergraduate and graduate statistics courses. "
        "The ideal candidate has strong teaching experience, a terminal degree "
        "in Statistics or a closely related field, and familiarity with modern "
        "educational technologies."
    )

    # Fields considered statistics-adjacent for degree evaluation
    statistics_adjacent_fields: list[str] = field(default_factory=lambda: [
        "Statistics",
        "Biostatistics",
        "Applied Statistics",
        "Mathematical Statistics",
        "Data Science",
        "Mathematics",
        "Applied Mathematics",
        "Operations Research",
        "Econometrics",
        "Quantitative Psychology",
        "Psychometrics",
        "Computer Science",
        "Machine Learning",
        "Actuarial Science",
        "Epidemiology",
        "Quantitative Finance",
    ])


@dataclass
class RatingRubric:
    """Star rating rubrics for teaching and research evaluation."""

    teaching_rubric: dict[int, str] = field(default_factory=lambda: {
        5: (
            "Extensive teaching experience (5+ courses as instructor), excellent "
            "evaluations mentioned, innovative pedagogy, AI/technology integration, "
            "statistics-specific courses, evidence of course development"
        ),
        4: (
            "Strong teaching record (3-5 courses), clear teaching philosophy, "
            "some technology integration, positive evaluations"
        ),
        3: (
            "Adequate experience (1-3 courses or significant TA experience), "
            "standard teaching statement, basic technology use"
        ),
        2: (
            "Limited teaching (TA only), generic teaching statement, "
            "no evidence of independent course instruction"
        ),
        1: (
            "No teaching experience, no teaching statement, "
            "or major concerns about teaching ability"
        ),
    })

    research_rubric: dict[int, str] = field(default_factory=lambda: {
        5: (
            "Active research agenda, publications in statistics journals, "
            "potential for collaboration with Purdue faculty, funded research"
        ),
        4: (
            "Some publications, clear research direction, "
            "statistics-relevant research area"
        ),
        3: (
            "Dissertation research only but reasonable potential, "
            "some conference presentations"
        ),
        2: "Minimal research activity, few or no publications",
        1: "No research or non-statistics research with no clear connection",
    })


@dataclass
class RedFlagCriteria:
    """Criteria for identifying red flags in applications."""
    checks: list[str] = field(default_factory=lambda: [
        "Degree not in Statistics or an adjacent field",
        "ABD status with no defense date or timeline",
        "Employment gaps greater than 1 year without explanation",
        "No teaching experience whatsoever",
        "Generic/boilerplate cover letter (not tailored to Purdue)",
        "Letters of recommendation that contradict CV claims",
        "Primarily research-focused with no teaching interest expressed",
        "Concerns raised in recommendation letters",
        "Mismatched dates or inconsistencies across documents",
        "No mention of statistics-relevant coursework or training",
    ])

    severity_levels: list[str] = field(default_factory=lambda: [
        "Minor",     # Worth noting but not disqualifying
        "Moderate",  # Warrants discussion by committee
        "Serious",   # Potentially disqualifying
    ])


@dataclass
class PositionWeights:
    """
    Controls which evaluation dimension is primary vs secondary.

    Drives:
    - Prompt emphasis ("Teaching is the PRIMARY criterion…")
    - Comparative ranking sort order
    - Combined-score weighting for tier assignment

    For a teaching-focused VAP, primary="teaching".
    For a tenure-track research position, primary="research".
    """
    primary: str = "teaching"      # "teaching" or "research"
    secondary: str = "research"    # the other one
    primary_weight: float = 2.0    # multiplier for combined score
    secondary_weight: float = 1.0


@dataclass
class ViewerConfig:
    """Web viewer settings."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True


@dataclass
class Config:
    """Master configuration combining all settings."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    weights: PositionWeights = field(default_factory=PositionWeights)
    rubric: RatingRubric = field(default_factory=RatingRubric)
    red_flags: RedFlagCriteria = field(default_factory=RedFlagCriteria)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    # File handling
    supported_extensions: list[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".doc", ".txt", ".rtf"
    ])

    # Paths (set dynamically based on CLI input)
    applicants_dir: Path = field(default_factory=lambda: Path("."))
    results_dir: Path = field(default_factory=lambda: Path("data/results"))

    def get_results_path(self, applicant_name: str) -> Path:
        """Get the results directory for a specific applicant."""
        path = self.results_dir / applicant_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Convenience helpers for prompt building ──

    def teaching_emphasis(self) -> str:
        """Return 'PRIMARY' or 'SECONDARY' for teaching."""
        return "PRIMARY" if self.weights.primary == "teaching" else "SECONDARY"

    def research_emphasis(self) -> str:
        """Return 'PRIMARY' or 'SECONDARY' for research."""
        return "PRIMARY" if self.weights.primary == "research" else "SECONDARY"

    def position_summary(self) -> str:
        """One-line position summary for prompt injection."""
        p = self.position
        parts = [p.position_title]
        if p.position_focus:
            parts.append(f"({p.position_focus})")
        if p.position_duration:
            parts.append(f"— {p.position_duration}")
        return " ".join(parts)

    def rubric_text(self, dimension: str) -> str:
        """
        Format a rubric dict as numbered text for prompt injection.

        Args:
            dimension: "teaching" or "research"
        """
        rubric_dict = (
            self.rubric.teaching_rubric
            if dimension == "teaching"
            else self.rubric.research_rubric
        )
        lines = []
        for stars in sorted(rubric_dict.keys(), reverse=True):
            desc = rubric_dict[stars].strip()
            lines.append(f"{stars} stars: {desc}")
        return "\n".join(lines)

    def red_flags_text(self) -> str:
        """Format the red flag checklist as numbered text for prompts."""
        lines = []
        for i, check in enumerate(self.red_flags.checks, 1):
            lines.append(f"{i}. {check}")
        return "\n".join(lines)

    def adjacent_fields_text(self) -> str:
        """Comma-separated list of adjacent fields for prompts."""
        return ", ".join(self.position.statistics_adjacent_fields)


# ── Global default config ──
DEFAULT_CONFIG = Config()


# ════════════════════════════════════════════════════════════════════════════
# YAML POSITION PROFILE LOADER
# ════════════════════════════════════════════════════════════════════════════


def load_position_yaml(yaml_path: str | Path) -> Config:
    """
    Load a position YAML profile and return a Config with those values applied.

    Only the position-specific sections are overridden — LLM, RAG, ensemble,
    and viewer settings keep their defaults (and can still be overridden by
    CLI flags or by editing this file).

    Args:
        yaml_path: Path to a position YAML file.

    Returns:
        A Config instance with position, weights, rubric, red_flags, and
        adjacent_fields populated from the YAML.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML is missing required sections.
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Position profile not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid YAML: {yaml_path}")

    config = Config()

    # ── Position details ──
    pos = data.get("position", {})
    if pos:
        config.position = PositionConfig(
            position_title=pos.get("title", config.position.position_title),
            position_focus=pos.get("focus", config.position.position_focus),
            position_duration=pos.get("duration", config.position.position_duration),
            department=pos.get("department", config.position.department),
            university=pos.get("university", config.position.university),
            description=pos.get("description", config.position.description),
            statistics_adjacent_fields=(
                data.get("adjacent_fields", config.position.statistics_adjacent_fields)
            ),
        )

    # ── Weights ──
    w = data.get("weights", {})
    if w:
        config.weights = PositionWeights(
            primary=w.get("primary", config.weights.primary),
            secondary=w.get("secondary", config.weights.secondary),
            primary_weight=float(w.get("primary_weight", config.weights.primary_weight)),
            secondary_weight=float(w.get("secondary_weight", config.weights.secondary_weight)),
        )

    # ── Rubrics ──
    rubric_data = data.get("rubric", {})
    if rubric_data:
        teaching_rubric = rubric_data.get("teaching")
        research_rubric = rubric_data.get("research")

        if teaching_rubric:
            # YAML keys might be ints or strings — normalise to int
            config.rubric.teaching_rubric = {
                int(k): str(v).strip() for k, v in teaching_rubric.items()
            }
        if research_rubric:
            config.rubric.research_rubric = {
                int(k): str(v).strip() for k, v in research_rubric.items()
            }

    # ── Red flags ──
    rf = data.get("red_flags", {})
    if rf:
        checks = rf.get("checks")
        if checks:
            config.red_flags.checks = checks
        severity = rf.get("severity_levels")
        if severity:
            config.red_flags.severity_levels = severity

    # ── Adjacent fields (also written into position) ──
    adj = data.get("adjacent_fields")
    if adj:
        config.position.statistics_adjacent_fields = adj

    return config