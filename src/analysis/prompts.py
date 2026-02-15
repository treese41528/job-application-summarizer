"""
LLM Prompt Templates.

All prompts used in the analysis pipeline. Centralized here for easy
tuning and iteration. Each prompt is designed to elicit structured JSON output.

Position-specific text (title, department, rubrics, red flags, etc.) is
injected at runtime via ``build_*()`` functions that accept a ``Config``
object.  Per-applicant placeholders (``{name}``, ``{degree_field}``, etc.)
remain as ``.format()`` targets for the evaluator.

Organization:
  CATEGORIZE_*             — Document classification (position-agnostic)
  EXTRACT_* / SUPPLEMENT_* — Profile building
  build_evaluate_*()       — Evaluation prompt builders (config-aware)
  build_rag_evaluate_*()   — RAG-grounded evaluation builders
  build_synthesize_*()     — Ensemble synthesizer prompt builders
  format_ensemble_results_for_synthesis()  — Helper to format model outputs

Backward compatibility:
  The old module-level constants (EVALUATE_TEACHING, RAG_EVALUATE_TEACHING,
  SYNTHESIZE_TEACHING, etc.) are still exported with hard-coded VAP defaults.
  New code should use the ``build_*()`` functions instead.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


# ════════════════════════════════════════════════════════════════════════════
# DOCUMENT CATEGORIZATION  (position-agnostic — no config needed)
# ════════════════════════════════════════════════════════════════════════════

CATEGORIZE_DOCUMENT = """You are classifying academic job application documents for a faculty position.

Analyze the following document and classify it into exactly ONE category:
- cv: Curriculum vitae listing education, employment history, publications, presentations, teaching
- cover_letter: Letter addressed to a hiring committee or search committee expressing interest in the position
- teaching_statement: Discusses teaching philosophy, pedagogical methods, and teaching experience in depth
- research_statement: Describes research agenda, past research accomplishments, and future research plans
- letter_of_recommendation: Written by a third party (not the candidate) evaluating or recommending the candidate
- other: Transcripts, diversity statements, writing samples, or anything that doesn't fit above

Key distinguishing features:
- A CV lists many items (education, jobs, publications) in a structured format
- A cover letter is addressed to someone and expresses interest in a specific position
- A teaching statement focuses on HOW the person teaches and their philosophy
- A research statement focuses on WHAT the person researches
- A recommendation letter is written ABOUT the candidate by someone else
- If a document discusses both teaching and research extensively but is structured as a letter to a committee, it's a cover_letter

Document text (first 3000 characters):
---
{document_text}
---

Respond with ONLY valid JSON, no other text:
{{"category": "one of: cv, cover_letter, teaching_statement, research_statement, letter_of_recommendation, other", "confidence": 0.0, "reasoning": "brief explanation"}}"""


# ════════════════════════════════════════════════════════════════════════════
# PROFILE EXTRACTION  (position-agnostic — no config needed)
# ════════════════════════════════════════════════════════════════════════════

EXTRACT_PROFILE_FROM_CV = """You are extracting structured information from an academic CV for a job application.

Extract the following information from this CV. If information is not present, use null.
Be precise with dates, counts, and names. For courses, include ALL courses listed.

CV Text:
---
{cv_text}
---

Respond with ONLY valid JSON matching this exact schema:
{{
  "name": "Full name of the candidate",
  "email": "email address or null",
  "current_position": "Current job title or null",
  "current_institution": "Current employer/university or null",
  "website": "Personal or academic website URL or null",

  "terminal_degree": "Highest degree with field (e.g., 'Ph.D. in Statistics')",
  "degree_field": "Field of study only (e.g., 'Statistics')",
  "degree_institution": "University where degree was/will be earned",
  "degree_year": null,
  "degree_status": "Completed or Expected Month Year or ABD",
  "additional_degrees": ["List of other degrees, e.g., 'M.S. in Mathematics, University of X, 2019'"],

  "courses_taught": [
    {{
      "course_name": "Course title",
      "course_number": "e.g., STAT 101 or null",
      "level": "undergraduate or graduate",
      "role": "instructor or ta or guest_lecturer or co-instructor",
      "institution": "University name",
      "semesters": 1
    }}
  ],

  "teaching_experience_years": 0.0,
  "teaching_technologies": ["R", "Python", "SAS", "JMP", "etc."],
  "ai_in_education": false,
  "ai_education_details": "Description of how they use AI in teaching, or null",

  "research_areas": ["Area 1", "Area 2"],
  "publications_count": 0,
  "grants_count": 0,
  "notable_publications": ["Up to 3 most notable publications"],

  "awards": ["List of awards and honors"],
  "professional_service": ["List of service activities"],
  "languages": ["Languages spoken"]
}}

IMPORTANT:
- degree_year should be an integer (year only) or null
- For degree_status, use exactly "Completed", "Expected Month Year", or "ABD"
- Count publications carefully from the publications section
- Include ALL courses, even if listed under teaching assistant experience
- teaching_experience_years should be approximate total years"""


# ════════════════════════════════════════════════════════════════════════════
# SUPPLEMENT FROM COVER LETTER
# ════════════════════════════════════════════════════════════════════════════

_SUPPLEMENT_TEMPLATE = """You are cross-referencing a cover letter against an existing applicant profile to find additional details.

The candidate's name is {name} and they are applying for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__.

Existing profile data:
{existing_profile_json}

Cover Letter Text:
---
{cover_letter_text}
---

Identify any NEW information from the cover letter that is NOT already in the profile. Focus on:
- Corrections to existing data
- Additional teaching experience or philosophy details
- Specific mention of __UNIVERSITY__ or why they want this position
- Technology or AI education mentions
- Any information about degree status or timeline

Respond with ONLY valid JSON:
{{
  "corrections": [
    {{"field": "field_name", "current_value": "what we have", "corrected_value": "what it should be", "reason": "why"}}
  ],
  "additional_teaching_technologies": ["any new technologies mentioned"],
  "ai_in_education": false,
  "ai_education_details": "new AI/education details or null",
  "university_specific_interest": "why they mention wanting __UNIVERSITY__ specifically, or null",
  "additional_notes": "any other important info not captured elsewhere"
}}"""


def build_supplement_from_cover_letter(config: "Config") -> str:
    """Build the cover-letter supplementation prompt template."""
    return _inject_position(config, _SUPPLEMENT_TEMPLATE)


# Backward-compatible default
SUPPLEMENT_PROFILE_FROM_COVER_LETTER = (
    _SUPPLEMENT_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus, 2-year term)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
)


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION — CLASSIC  (text-stuffed, no RAG)
# ════════════════════════════════════════════════════════════════════════════

_EVALUATE_TEACHING_TEMPLATE = """You are evaluating a candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. This is a __DURATION__ role where teaching is the __TEACHING_EMPHASIS__ criterion.

Candidate: {name}
Terminal Degree: {terminal_degree} ({degree_status})
Degree Field: {degree_field}

Teaching-related documents:
---
CV TEACHING SECTIONS:
{cv_teaching_text}

TEACHING STATEMENT:
{teaching_statement_text}
---

RATING RUBRIC:
__TEACHING_RUBRIC__

Evaluate this candidate's teaching qualifications. Be specific and cite evidence from the documents.

Respond with ONLY valid JSON:
{{
  "teaching_stars": 0,
  "teaching_strengths": ["specific strength with evidence"],
  "teaching_weaknesses": ["specific weakness or gap"],
  "teaching_evidence": ["direct factual evidence from documents supporting the rating"]
}}"""


_EVALUATE_RESEARCH_TEMPLATE = """You are evaluating the research profile of a candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. Research is the __RESEARCH_EMPHASIS__ criterion for this role.

Candidate: {name}
Degree Field: {degree_field}
Research Areas: {research_areas}
Publications: {publications_count}

RESEARCH STATEMENT:
{research_statement_text}

RATING RUBRIC:
__RESEARCH_RUBRIC__

Respond with ONLY valid JSON:
{{
  "research_stars": 0,
  "research_strengths": ["specific strength"],
  "research_weaknesses": ["specific weakness or gap"]
}}"""


_EVALUATE_FIT_TEMPLATE = """You are performing a holistic fit assessment and red flag check for a candidate applying to be a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__.

CANDIDATE PROFILE:
{profile_json}

ALL DOCUMENT SUMMARIES:
{all_documents_summary}

POSITION REQUIREMENTS:
__POSITION_DESCRIPTION__
- Terminal degree in a relevant field required
- __ADJACENT_FIELD_LABEL__: __ADJACENT_FIELDS__

RED FLAG CHECKLIST - check each one:
__RED_FLAG_CHECKS__

Respond with ONLY valid JSON:
{{
  "fit_score": 0,
  "fit_rationale": "2-3 sentence explanation of fit with this role",
  "red_flags": [
    {{
      "category": "e.g., Degree Mismatch",
      "severity": "Minor or Moderate or Serious",
      "description": "Specific explanation",
      "source_document": "which document this was found in"
    }}
  ],
  "overall_recommendation": "Strong or Moderate or Weak or Do Not Advance",
  "executive_summary": "2-3 sentence overall assessment suitable for a search committee"
}}"""


_SUMMARIZE_LETTER_TEMPLATE = """You are summarizing a letter of recommendation for a candidate applying to a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__.

Candidate Name: {candidate_name}

Letter Text:
---
{letter_text}
---

Extract key information about this recommendation letter. Pay special attention to comments about __PRIMARY_FOCUS__ ability since this is a __PRIMARY_FOCUS__-focused position.

Respond with ONLY valid JSON:
{{
  "writer_name": "Name of the letter writer or null if unclear",
  "writer_title": "Title/position of the writer or null",
  "writer_institution": "Institution of the writer or null",
  "relationship_to_candidate": "advisor, teaching mentor, colleague, department chair, etc.",
  "tone": "Highly Positive, Positive, Mixed, Tepid, or Negative",
  "key_points": ["up to 5 key points made in the letter"],
  "teaching_comments": "Summary of what the letter says about teaching ability, or null",
  "research_comments": "Summary of what the letter says about research, or null",
  "concerns_raised": "Any concerns or reservations expressed, or null"
}}"""


# ── Builder functions (config-aware) ──

def build_evaluate_teaching(config: "Config") -> str:
    return _inject_position(config, _EVALUATE_TEACHING_TEMPLATE)

def build_evaluate_research(config: "Config") -> str:
    return _inject_position(config, _EVALUATE_RESEARCH_TEMPLATE)

def build_evaluate_fit_and_flags(config: "Config") -> str:
    return _inject_position(config, _EVALUATE_FIT_TEMPLATE)

def build_summarize_letter(config: "Config") -> str:
    return _inject_position(config, _SUMMARIZE_LETTER_TEMPLATE)


# ── Backward-compatible constants (default VAP teaching config) ──

EVALUATE_TEACHING = (
    _EVALUATE_TEACHING_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__DURATION__", "temporary 2-year")
    .replace("__TEACHING_EMPHASIS__", "PRIMARY")
    .replace("__TEACHING_RUBRIC__",
        "5 stars: Extensive teaching experience (5+ courses as instructor), excellent evaluations mentioned, innovative pedagogy, AI/technology integration, statistics-specific courses, evidence of course development\n"
        "4 stars: Strong teaching record (3-5 courses), clear teaching philosophy, some technology integration, positive evaluations\n"
        "3 stars: Adequate experience (1-3 courses or significant TA experience), standard teaching statement, basic technology use\n"
        "2 stars: Limited teaching (TA only), generic teaching statement, no evidence of independent course instruction\n"
        "1 star: No teaching experience, no teaching statement, or major concerns about teaching ability")
)

EVALUATE_RESEARCH = (
    _EVALUATE_RESEARCH_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__RESEARCH_EMPHASIS__", "SECONDARY")
    .replace("__RESEARCH_RUBRIC__",
        "5 stars: Active research agenda, publications in statistics journals, potential for collaboration with Purdue faculty, funded research\n"
        "4 stars: Some publications, clear research direction, statistics-relevant research area\n"
        "3 stars: Dissertation research only but reasonable potential, some conference presentations\n"
        "2 stars: Minimal research activity, few or no publications\n"
        "1 star: No research or non-statistics research with no clear connection")
)

EVALUATE_FIT_AND_FLAGS = (
    _EVALUATE_FIT_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus, 2-year term)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__POSITION_DESCRIPTION__",
        "- Primary focus: Teaching undergraduate and graduate statistics courses\n"
        "- 2-year temporary appointment\n"
        "- Strong teaching record expected")
    .replace("__ADJACENT_FIELD_LABEL__", "Statistics-adjacent fields")
    .replace("__ADJACENT_FIELDS__",
        "Statistics, Biostatistics, Applied Statistics, Mathematical Statistics, "
        "Data Science, Mathematics, Applied Mathematics, Operations Research, "
        "Econometrics, Quantitative Psychology, Psychometrics, Computer Science, "
        "Machine Learning, Actuarial Science, Epidemiology, Quantitative Finance")
    .replace("__RED_FLAG_CHECKS__",
        "1. Degree not in Statistics or an adjacent field listed above\n"
        "2. ABD status with no defense date or timeline\n"
        "3. Employment gaps greater than 1 year without explanation\n"
        "4. No teaching experience whatsoever\n"
        "5. Generic/boilerplate cover letter (not tailored to Purdue)\n"
        "6. Letters that contradict CV claims\n"
        "7. Primarily research-focused with no teaching interest expressed\n"
        "8. Concerns raised in recommendation letters\n"
        "9. Mismatched dates or inconsistencies across documents\n"
        "10. No mention of statistics-relevant coursework or training")
)

SUMMARIZE_RECOMMENDATION_LETTER = (
    _SUMMARIZE_LETTER_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__PRIMARY_FOCUS__", "teaching")
)


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION — RAG-GROUNDED
# ════════════════════════════════════════════════════════════════════════════

_RAG_EVALUATE_TEACHING_TEMPLATE = """You are evaluating a candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. Teaching is the __TEACHING_EMPHASIS__ criterion.

You have access to the candidate's full application documents in your knowledge base, including their CV, teaching statement, cover letter, and recommendation letters. Search through these documents to find all teaching-related evidence.

CANDIDATE SUMMARY:
- Name: {name}
- Terminal Degree: {terminal_degree} ({degree_status})
- Degree Field: {degree_field}
- Courses as Instructor: {instructor_course_count}
- Courses as TA: {ta_course_count}
- Teaching Technologies: {teaching_technologies}
- AI in Education: {ai_in_education}

RATING RUBRIC:
__TEACHING_RUBRIC__

INSTRUCTIONS:
- Search the candidate's documents for ALL evidence of teaching experience
- Look for specific course names, enrollment numbers, evaluation scores, teaching awards
- Check for innovative methods, technology use, curriculum development
- Note any teaching philosophy or pedagogical approach described
- Be specific and cite evidence from the documents

Respond with ONLY valid JSON:
{{
  "teaching_stars": 0,
  "teaching_strengths": ["specific strength with evidence from documents"],
  "teaching_weaknesses": ["specific weakness or gap"],
  "teaching_evidence": ["direct factual evidence found in documents supporting the rating"]
}}"""


_RAG_EVALUATE_RESEARCH_TEMPLATE = """You are evaluating the research profile of a candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. Research is the __RESEARCH_EMPHASIS__ criterion for this role.

You have access to the candidate's full application documents in your knowledge base. Search for research-related evidence across all documents.

CANDIDATE SUMMARY:
- Name: {name}
- Degree Field: {degree_field}
- Research Areas: {research_areas}
- Publications Count: {publications_count}

RATING RUBRIC:
__RESEARCH_RUBRIC__

INSTRUCTIONS:
- Search for the research statement and CV publication list
- Count actual publications (distinguish journal articles, conference papers, preprints)
- Note any grants, funding, or collaboration potential
- Assess relevance to the department and position

Respond with ONLY valid JSON:
{{
  "research_stars": 0,
  "research_strengths": ["specific strength with evidence"],
  "research_weaknesses": ["specific weakness or gap"]
}}"""


_RAG_EVALUATE_FIT_TEMPLATE = """You are performing a holistic fit assessment and red flag check for a candidate applying to be a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__.

You have access to the candidate's full application documents in your knowledge base. Search thoroughly across ALL documents to check for red flags and assess fit.

CANDIDATE PROFILE:
{profile_json}

POSITION REQUIREMENTS:
__POSITION_DESCRIPTION__
- Terminal degree in a relevant field required
- __ADJACENT_FIELD_LABEL__: __ADJACENT_FIELDS__

RED FLAG CHECKLIST — search documents for evidence of each:
__RED_FLAG_CHECKS__

INSTRUCTIONS:
- Cross-reference claims across CV, cover letter, teaching statement, and letters
- Look for inconsistencies in dates, roles, or claims
- Check if the cover letter mentions __UNIVERSITY__ specifically
- Read recommendation letters for any reservations or lukewarm language
- Assess overall fit for this position

Respond with ONLY valid JSON:
{{
  "fit_score": 0,
  "fit_rationale": "2-3 sentence explanation of fit with this role",
  "red_flags": [
    {{
      "category": "e.g., Degree Mismatch",
      "severity": "Minor or Moderate or Serious",
      "description": "Specific explanation with evidence from documents",
      "source_document": "which document this was found in"
    }}
  ],
  "overall_recommendation": "Strong or Moderate or Weak or Do Not Advance",
  "executive_summary": "2-3 sentence overall assessment suitable for a search committee"
}}"""


_RAG_SUMMARIZE_LETTER_TEMPLATE = """You are summarizing a letter of recommendation for a candidate applying to a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__.

Candidate Name: {candidate_name}
Letter filename: {letter_filename}

You have access to the candidate's documents in your knowledge base. Search for the letter of recommendation from the file named above and summarize it.

Pay special attention to comments about __PRIMARY_FOCUS__ ability since this is a __PRIMARY_FOCUS__-focused position.

Respond with ONLY valid JSON:
{{
  "writer_name": "Name of the letter writer or null if unclear",
  "writer_title": "Title/position of the writer or null",
  "writer_institution": "Institution of the writer or null",
  "relationship_to_candidate": "advisor, teaching mentor, colleague, department chair, etc.",
  "tone": "Highly Positive, Positive, Mixed, Tepid, or Negative",
  "key_points": ["up to 5 key points made in the letter"],
  "teaching_comments": "Summary of what the letter says about teaching ability, or null",
  "research_comments": "Summary of what the letter says about research, or null",
  "concerns_raised": "Any concerns or reservations expressed, or null"
}}"""


# ── RAG builder functions ──

def build_rag_evaluate_teaching(config: "Config") -> str:
    return _inject_position(config, _RAG_EVALUATE_TEACHING_TEMPLATE)

def build_rag_evaluate_research(config: "Config") -> str:
    return _inject_position(config, _RAG_EVALUATE_RESEARCH_TEMPLATE)

def build_rag_evaluate_fit_and_flags(config: "Config") -> str:
    return _inject_position(config, _RAG_EVALUATE_FIT_TEMPLATE)

def build_rag_summarize_letter(config: "Config") -> str:
    return _inject_position(config, _RAG_SUMMARIZE_LETTER_TEMPLATE)


# ── Backward-compatible RAG constants ──

RAG_EVALUATE_TEACHING = (
    _RAG_EVALUATE_TEACHING_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus, 2-year term)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__TEACHING_EMPHASIS__", "PRIMARY")
    .replace("__TEACHING_RUBRIC__",
        "5 stars: Extensive teaching experience (5+ courses as instructor), excellent evaluations, innovative pedagogy, AI/technology integration, statistics-specific courses, course development\n"
        "4 stars: Strong teaching record (3-5 courses), clear philosophy, some technology integration, positive evaluations\n"
        "3 stars: Adequate experience (1-3 courses or significant TA), standard statement, basic technology\n"
        "2 stars: Limited teaching (TA only), generic statement, no independent instruction\n"
        "1 star: No teaching experience, no teaching statement, or major concerns")
)

RAG_EVALUATE_RESEARCH = (
    _RAG_EVALUATE_RESEARCH_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__RESEARCH_EMPHASIS__", "SECONDARY")
    .replace("__RESEARCH_RUBRIC__",
        "5 stars: Active research agenda, publications in statistics journals, potential for collaboration, funded research\n"
        "4 stars: Some publications, clear direction, statistics-relevant area\n"
        "3 stars: Dissertation research only but reasonable potential, conference presentations\n"
        "2 stars: Minimal activity, few or no publications\n"
        "1 star: No research or non-statistics research with no clear connection")
)

RAG_EVALUATE_FIT_AND_FLAGS = (
    _RAG_EVALUATE_FIT_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus, 2-year term)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__POSITION_DESCRIPTION__",
        "- Primary focus: Teaching undergraduate and graduate statistics courses\n"
        "- 2-year temporary appointment\n"
        "- Strong teaching record expected")
    .replace("__ADJACENT_FIELD_LABEL__", "Statistics-adjacent fields")
    .replace("__ADJACENT_FIELDS__",
        "Statistics, Biostatistics, Applied Statistics, Mathematical Statistics, "
        "Data Science, Mathematics, Applied Mathematics, Operations Research, "
        "Econometrics, Quantitative Psychology, Psychometrics, Computer Science, "
        "Machine Learning, Actuarial Science, Epidemiology, Quantitative Finance")
    .replace("__RED_FLAG_CHECKS__",
        "1. Degree not in Statistics or an adjacent field listed above\n"
        "2. ABD status with no defense date or timeline\n"
        "3. Employment gaps greater than 1 year without explanation\n"
        "4. No teaching experience whatsoever\n"
        "5. Generic/boilerplate cover letter (not tailored to Purdue)\n"
        "6. Letters that contradict CV claims\n"
        "7. Primarily research-focused with no teaching interest expressed\n"
        "8. Concerns raised in recommendation letters\n"
        "9. Mismatched dates or inconsistencies across documents\n"
        "10. No mention of statistics-relevant coursework or training")
)

RAG_SUMMARIZE_RECOMMENDATION_LETTER = (
    _RAG_SUMMARIZE_LETTER_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__PRIMARY_FOCUS__", "teaching")
)


# ════════════════════════════════════════════════════════════════════════════
# ENSEMBLE SYNTHESIS PROMPTS
# ════════════════════════════════════════════════════════════════════════════

_SYNTHESIZE_TEACHING_TEMPLATE = """You are a senior faculty search committee member synthesizing multiple independent evaluations of a candidate's teaching qualifications.

{num_evaluations} different evaluators independently assessed this candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. Each used the same rubric but may have weighted evidence differently.

CANDIDATE: {name}

INDIVIDUAL EVALUATIONS:
{evaluations_text}

YOUR TASK:
Synthesize these evaluations into ONE final teaching assessment. You should:
1. If ratings vary, determine the most justified rating based on the evidence cited
2. Merge strengths — keep all unique strengths, remove duplicates
3. Merge weaknesses — keep all unique weaknesses, remove duplicates
4. Combine evidence — the final list should include the best citations from all evaluators
5. If evaluators disagree on a point, note the disagreement in evidence

IMPORTANT:
- Do NOT simply average the star ratings. Weigh the quality of evidence behind each.
- If one evaluator cites strong concrete evidence (course names, evaluation scores) and another gives a rating without evidence, trust the evidence-backed rating.
- A rating that 2 out of 3 evaluators agree on carries more weight than an outlier.

Respond with ONLY valid JSON:
{{
  "teaching_stars": 0,
  "teaching_strengths": ["merged strengths with best evidence"],
  "teaching_weaknesses": ["merged weaknesses"],
  "teaching_evidence": ["combined evidence from all evaluators"],
  "synthesis_notes": "Brief explanation of how you resolved any disagreements between evaluators"
}}"""


_SYNTHESIZE_RESEARCH_TEMPLATE = """You are a senior faculty search committee member synthesizing multiple independent evaluations of a candidate's research qualifications.

{num_evaluations} different evaluators independently assessed this candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. Research is the __RESEARCH_EMPHASIS__ criterion for this role.

CANDIDATE: {name}

INDIVIDUAL EVALUATIONS:
{evaluations_text}

YOUR TASK:
Synthesize into ONE final research assessment:
1. Determine the most justified rating based on cited evidence
2. Merge unique strengths and weaknesses
3. If evaluators disagree, note it

Respond with ONLY valid JSON:
{{
  "research_stars": 0,
  "research_strengths": ["merged strengths"],
  "research_weaknesses": ["merged weaknesses"],
  "synthesis_notes": "Brief explanation of how disagreements were resolved"
}}"""


_SYNTHESIZE_FIT_TEMPLATE = """You are a senior faculty search committee member synthesizing multiple independent fit assessments and red flag analyses.

{num_evaluations} different evaluators independently assessed this candidate for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__.

CANDIDATE: {name}

TEACHING ASSESSMENT (already synthesized):
- Teaching Stars: {teaching_stars}/5
- Key Strengths: {teaching_strengths}

RESEARCH ASSESSMENT (already synthesized):
- Research Stars: {research_stars}/5

LETTER CONSENSUS: {letter_consensus}

INDIVIDUAL FIT & FLAG EVALUATIONS:
{evaluations_text}

YOUR TASK:
Synthesize into ONE final fit assessment:
1. Determine the most justified fit score (1-5)
2. Merge all unique red flags. If multiple evaluators found the same flag, keep it once with the highest severity.
3. Remove spurious red flags — if only one evaluator flagged something and the evidence is weak, you may discard it.
4. Determine the overall recommendation considering the already-synthesized teaching and research scores.
5. Write an executive summary that a search committee chair could read in 30 seconds.

RECOMMENDATION LEVELS:
- "Strong": Clearly should advance. Strong in __PRIMARY_FOCUS__, reasonable in __SECONDARY_FOCUS__, good fit.
- "Moderate": Worth discussing. Some strengths, some gaps.
- "Weak": Notable concerns. Below average for the pool.
- "Do Not Advance": Disqualifying issues or poor fit.

Respond with ONLY valid JSON:
{{
  "fit_score": 0,
  "fit_rationale": "2-3 sentence explanation",
  "red_flags": [
    {{
      "category": "e.g., Degree Mismatch",
      "severity": "Minor or Moderate or Serious",
      "description": "Explanation with evidence",
      "source_document": "which document"
    }}
  ],
  "overall_recommendation": "Strong or Moderate or Weak or Do Not Advance",
  "executive_summary": "2-3 sentence overall assessment suitable for a search committee chair"
}}"""


# ── Synthesis builder functions ──

def build_synthesize_teaching(config: "Config") -> str:
    return _inject_position(config, _SYNTHESIZE_TEACHING_TEMPLATE)

def build_synthesize_research(config: "Config") -> str:
    return _inject_position(config, _SYNTHESIZE_RESEARCH_TEMPLATE)

def build_synthesize_fit_and_flags(config: "Config") -> str:
    return _inject_position(config, _SYNTHESIZE_FIT_TEMPLATE)


# ── Backward-compatible synthesis constants ──

SYNTHESIZE_TEACHING = (
    _SYNTHESIZE_TEACHING_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
)

SYNTHESIZE_RESEARCH = (
    _SYNTHESIZE_RESEARCH_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__RESEARCH_EMPHASIS__", "SECONDARY")
)

SYNTHESIZE_FIT_AND_FLAGS = (
    _SYNTHESIZE_FIT_TEMPLATE
    .replace("__POSITION_SUMMARY__", "Visiting Assistant Professor (Teaching Focus, 2-year term)")
    .replace("__DEPARTMENT__", "the Department of Statistics")
    .replace("__UNIVERSITY__", "Purdue University")
    .replace("__PRIMARY_FOCUS__", "teaching")
    .replace("__SECONDARY_FOCUS__", "research")
)


# ════════════════════════════════════════════════════════════════════════════
# COMPARATIVE RANKING PROMPT (config-aware)
# ════════════════════════════════════════════════════════════════════════════

_COMPARATIVE_RANKING_TEMPLATE = """You are a faculty search committee assistant for a __POSITION_SUMMARY__ in __DEPARTMENT__ at __UNIVERSITY__. __PRIMARY_FOCUS_UPPER__ is the primary evaluation criterion.

Below is a summary of all candidates who have been individually evaluated. Your job is to COMPARATIVELY rank them against each other — not in isolation, but relative to this specific applicant pool.

{table}

For EACH candidate, provide:
1. __PRIMARY_FOCUS___rank: Integer rank for __PRIMARY_FOCUS__ (1 = strongest in the pool)
2. __SECONDARY_FOCUS___rank: Integer rank for __SECONDARY_FOCUS__ (1 = strongest in the pool)
3. __PRIMARY_FOCUS___curved: A 1-5 star rating ON A CURVE relative to this pool
   - 5 = top ~10% of THIS pool
   - 4 = above average in THIS pool
   - 3 = average for THIS pool
   - 2 = below average in THIS pool
   - 1 = weakest in THIS pool
4. __SECONDARY_FOCUS___curved: Same curve logic
5. tier: One of "Top", "Strong", "Middle", "Below Average", "Weak"
   - "Top" = clearly should advance to interview (top ~15%)
   - "Strong" = deserves serious consideration
   - "Middle" = solid but not standout
   - "Below Average" = notable gaps relative to pool
   - "Weak" = should not advance
6. comparative_notes: 1-2 sentences explaining how this candidate compares to others.
   Focus on what differentiates them — don't repeat the individual evaluation.

IMPORTANT:
- Rankings must be unique integers (no ties for rank)
- Curved scores CAN have ties (multiple candidates can be 4/5)
- Weight __PRIMARY_FOCUS__ MORE than __SECONDARY_FOCUS__
- Consider degree field fit for the department
- A candidate with 4/5 individual rating but less experience than others in the pool might get curved to 3/5

Respond with ONLY a JSON array, one object per candidate:
[
  {{
    "folder_name": "exact folder name from the table",
    "name": "candidate name",
    "teaching_rank": 1,
    "research_rank": 3,
    "teaching_curved": 5,
    "research_curved": 3,
    "tier": "Top",
    "comparative_notes": "Strongest profile in the pool with..."
  }},
  ...
]"""


def build_comparative_ranking_prompt(config: "Config") -> str:
    """Build the comparative ranking prompt template."""
    prompt = _inject_position(config, _COMPARATIVE_RANKING_TEMPLATE)
    # Additional marker for uppercase primary focus
    prompt = prompt.replace(
        "__PRIMARY_FOCUS_UPPER__",
        config.weights.primary.capitalize(),
    )
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# HELPER — FORMAT ENSEMBLE RESULTS FOR SYNTHESIS
# ════════════════════════════════════════════════════════════════════════════

def format_ensemble_results_for_synthesis(
    results: list[dict],
    key_fields: list[str] | None = None,
) -> str:
    """
    Format a list of ensemble model results into a readable text block
    for inclusion in a synthesis prompt.

    Args:
        results: List of dicts from client.ensemble_query_json(), each
                 containing "model", "data", and "error" keys.
        key_fields: If provided, only include these fields from each
                    result's data dict (reduces noise in the prompt).

    Returns:
        Formatted text with each evaluator's output clearly labelled.
    """
    import json as _json

    lines = []
    for i, result in enumerate(results, 1):
        model = result.get("model", "Unknown")
        error = result.get("error")
        data = result.get("data", {})

        lines.append(f"--- Evaluator {i} (Model: {model}) ---")

        if error:
            lines.append(f"  [EVALUATION FAILED: {error}]")
        elif data:
            display = (
                {k: data[k] for k in key_fields if k in data}
                if key_fields
                else data
            )
            lines.append(_json.dumps(display, indent=2))
        else:
            lines.append("  [No output]")

        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL — POSITION INJECTION
# ════════════════════════════════════════════════════════════════════════════


def _inject_position(config: "Config", template: str) -> str:
    """
    Replace all ``__MARKER__`` placeholders in a prompt template with
    values derived from the given Config.

    Markers:
        __POSITION_SUMMARY__     e.g. "Assistant Professor (Research) — Tenure-track"
        __DEPARTMENT__           e.g. "the Department of Statistics"
        __UNIVERSITY__           e.g. "Purdue University"
        __DURATION__             e.g. "tenure-track" or "temporary 2-year"
        __TEACHING_EMPHASIS__    "PRIMARY" or "SECONDARY"
        __RESEARCH_EMPHASIS__    "PRIMARY" or "SECONDARY"
        __TEACHING_RUBRIC__      Multi-line rubric text for teaching
        __RESEARCH_RUBRIC__      Multi-line rubric text for research
        __POSITION_DESCRIPTION__ Multi-line position requirements
        __ADJACENT_FIELD_LABEL__ Label like "Adjacent fields"
        __ADJACENT_FIELDS__      Comma-separated field list
        __RED_FLAG_CHECKS__      Numbered red flag checklist
        __PRIMARY_FOCUS__        "teaching" or "research"
        __SECONDARY_FOCUS__      the other one

    Per-applicant placeholders like ``{name}`` are left intact for
    the caller to fill via ``.format()``.
    """
    pos = config.position
    w = config.weights

    # Build position description
    desc_lines = []
    if pos.position_focus:
        desc_lines.append(f"- Primary focus: {pos.position_focus}")
    if pos.position_duration:
        desc_lines.append(f"- Duration: {pos.position_duration}")
    if pos.description:
        first_sentence = pos.description.strip().split(".")[0].strip()
        desc_lines.append(f"- {first_sentence}")
    position_description = "\n".join(desc_lines) if desc_lines else (pos.description or "")

    replacements = {
        "__POSITION_SUMMARY__": config.position_summary(),
        "__DEPARTMENT__": pos.department,
        "__UNIVERSITY__": pos.university,
        "__DURATION__": pos.position_duration or "",
        "__TEACHING_EMPHASIS__": config.teaching_emphasis(),
        "__RESEARCH_EMPHASIS__": config.research_emphasis(),
        "__TEACHING_RUBRIC__": config.rubric_text("teaching"),
        "__RESEARCH_RUBRIC__": config.rubric_text("research"),
        "__POSITION_DESCRIPTION__": position_description,
        "__ADJACENT_FIELD_LABEL__": "Adjacent fields",
        "__ADJACENT_FIELDS__": config.adjacent_fields_text(),
        "__RED_FLAG_CHECKS__": config.red_flags_text(),
        "__PRIMARY_FOCUS__": w.primary,
        "__SECONDARY_FOCUS__": w.secondary,
    }

    result = template
    for marker, value in replacements.items():
        result = result.replace(marker, value)

    return result