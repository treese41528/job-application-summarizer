# Job Application Summarizer

An LLM-powered pipeline that processes faculty job applications — configurable for any academic position type via YAML position profiles. It extracts text from PDF/DOCX documents, uses **Purdue GenAI Studio** LLMs to categorize, profile, and evaluate applicants, then ranks the pool comparatively and presents results in a local web viewer with a stage-based hiring pipeline and chat interface for follow-up questions against each applicant's documents.

Default configuration targets a **Visiting Assistant Professor (Teaching Focus, 2-year)** position in the Department of Statistics at Purdue University, but position profiles can be swapped via `--position` for tenure-track, research-focused, lecturer, or other searches.

> 📖 **[Full Technical Documentation](https://htmlpreview.github.io/?https://github.com/treese41528/job-application-summarizer/blob/main/docs/DOCUMENTATION.html)** — detailed pipeline walkthrough, architecture diagrams, rating rubrics, and troubleshooting guide

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Position Profiles](#position-profiles)
- [Processing Pipeline](#processing-pipeline)
- [Evaluation Modes](#evaluation-modes)
- [Stage Pipeline](#stage-pipeline)
- [CLI Reference](#cli-reference)
- [Web Viewer](#web-viewer)
- [Human Review Overlay](#human-review-overlay)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [RAG Knowledge Base Cleanup](#rag-knowledge-base-cleanup)
- [Evaluation Criteria](#evaluation-criteria)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your GenAI Studio API key
export GENAI_STUDIO_API_KEY="sk-your-key-here"

# 3. Copy the GenAI Studio SDK into the project root
cp /path/to/genai_studio.py .

# 4. Process applications (RAG enabled by default)
python run.py process ../vap-search-2025/

# 5. Launch the viewer
python run.py serve --applicants-dir ../vap-search-2025/
# Open http://127.0.0.1:5000
```

### Minimal run (no RAG, no ensemble)

```bash
python run.py process ../vap-search-2025/ --no-rag --no-ensemble --model gemma3:12b
```

### Different position type

```bash
# Tenure-track research-focused search
python run.py process ../tt-search-2026/ --position positions/tt-research-2026.yaml

# Re-rank with ensemble consensus
python run.py rank --ensemble --position positions/tt-research-2026.yaml
```

---

## Architecture

### Directory Layout

```
your-working-directory/
├── job-application-summarizer/         # This tool
│   ├── run.py                          # CLI entry point (orchestrator)
│   ├── genai_studio.py                 # Purdue GenAI Studio SDK (copy here)
│   ├── cleanup_rag.py                  # Standalone KB teardown script
│   ├── fix_paths.py                    # Fix document paths after moving machines
│   ├── requirements.txt
│   ├── positions/                      # Position profile YAML files
│   │   ├── vap-teaching-2025.yaml      # VAP teaching-focused (default config)
│   │   └── tt-research-2026.yaml       # Tenure-track research-focused example
│   ├── src/
│   │   ├── config.py                   # All configuration dataclasses + YAML loader
│   │   ├── llm/
│   │   │   └── client.py              # LLMClient (query, query_json, ensemble)
│   │   ├── rag/
│   │   │   └── manager.py            # ApplicantKBManager (upload, index, cleanup)
│   │   ├── models/
│   │   │   ├── applicant.py           # ApplicantProfile, CourseTaught
│   │   │   ├── document.py            # Document, DocumentCategory
│   │   │   └── evaluation.py          # Evaluation, RedFlag, LetterSummary
│   │   ├── extraction/
│   │   │   ├── pdf_extractor.py       # pdfplumber → pypdf fallback
│   │   │   ├── docx_extractor.py      # pandoc → python-docx, .doc via LibreOffice
│   │   │   └── text_cleaner.py        # clean_text(), chunk_text()
│   │   ├── analysis/
│   │   │   ├── categorizer.py         # Filename hints + LLM classification
│   │   │   ├── profile_builder.py     # Structured profile from CV + cover letter
│   │   │   ├── evaluator.py           # Teaching/research/fit evaluation
│   │   │   ├── prompts.py             # All prompt templates (classic, RAG, synthesis)
│   │   │   └── comparative.py         # Cross-pool ranking with curved scores + ensemble
│   │   └── viewer/
│   │       ├── app.py                 # Flask web app (stages, chat, review API)
│   │       ├── templates/             # dashboard.html, applicant.html, base.html
│   │       └── static/               # styles.css, app.js
│   ├── data/
│   │   └── results/                    # Generated JSON output per applicant
│   └── tests/
│
└── vap-search-2025/                    # Your applicant folders (separate)
    ├── Alice_Smith/
    │   ├── CV.pdf
    │   ├── CoverLetter.docx
    │   ├── TeachingStatement.pdf
    │   ├── ResearchStatement.pdf
    │   └── Letter_Doe.pdf
    └── Bob_Jones/
        └── ...
```

### How Requests Flow

| Operation | Route | Why |
|-----------|-------|-----|
| Document categorization | LLM `query_json()` | Filename hints + LLM fallback for ambiguous docs |
| Profile extraction | LLM `query_json()` | Structured JSON from CV text |
| Evaluation (classic) | LLM `query_json()` | Text-stuffed prompt with truncated docs |
| Evaluation (RAG) | LLM `query_json()` with `collections=[kb_id]` | Full-document retrieval, no truncation |
| Evaluation (ensemble) | Multiple LLMs → synthesizer | Each model evaluates independently, then merge |
| Comparative ranking | LLM `query_json()` | Cross-pool comparison table |
| Comparative ranking (ensemble) | Multiple LLMs → synthesizer | Each model ranks independently, then consensus |
| Chat interface | LLM `chat()` with `collections=[kb_id]` | Real-time Q&A against applicant's documents |
| Stage management | Viewer API | Committee-driven pipeline progression |

---

## Position Profiles

Instead of editing Python code to change position details, create a YAML file that defines everything position-specific. The `--position` flag is available on `process`, `rank`, and `serve` commands.

### YAML Format

```yaml
position:
  title: "Assistant Professor"
  focus: "Research (Primary)"
  duration: "Tenure-track"
  department: "Department of Statistics"
  university: "Purdue University"
  description: "A tenure-track position with emphasis on independent research..."

weights:
  primary: research          # "teaching" or "research"
  secondary: teaching
  primary_weight: 2.0        # multiplier for combined tier scoring
  secondary_weight: 1.0

rubric:
  teaching:
    5: "Description of 5-star teaching..."
    4: "..."
    3: "..."
    2: "..."
    1: "..."
  research:
    5: "Description of 5-star research..."
    4: "..."
    3: "..."
    2: "..."
    1: "..."

red_flags:
  checks:
    - "Degree not in Statistics or adjacent field"
    - "ABD with no defense timeline"
    - "No first-author publications"

adjacent_fields:
  - Statistics
  - Biostatistics
  - Data Science
```

### Included Profiles

| File | Position Type | Primary | Secondary |
|------|--------------|---------|-----------|
| `positions/vap-teaching-2025.yaml` | Visiting Assistant Professor | Teaching | Research |
| `positions/tt-research-2026.yaml` | Assistant Professor (Tenure-Track) | Research | Teaching |

### What the Profile Controls

| Aspect | What changes |
|--------|-------------|
| Prompt text | Position title, department, university injected into all evaluation prompts |
| Rating rubrics | 1–5 star criteria for both teaching and research |
| Red flag checklist | Which checks are performed during fit assessment |
| Ranking sort order | Primary dimension ranked first, weighted more heavily |
| Weight multiplier | Configurable ratio for combined tier scoring (default 2:1) |
| Chat system prompt | Viewer chat references the correct position context |
| Adjacent fields | Which degree fields are considered relevant |

### Without `--position`

Everything works exactly as before — hardcoded defaults match the VAP teaching position at Purdue Statistics. The `--position` flag is entirely opt-in.

---

## Processing Pipeline

```
Applicant Folder
     │
     ▼
 ① Extract         PDF (pdfplumber → pypdf) and DOCX (pandoc → python-docx) to plain text
     │
     ▼
 ② Categorize      Classify each document: CV, Cover Letter, Teaching Statement,
     │              Research Statement, Letter of Recommendation, Other
     ▼              (filename hints first; LLM fallback for ambiguous files)
 ③ RAG Index        Upload documents → create per-applicant knowledge base → wait for indexing
     │              (optional — skipped with --no-rag)
     ▼
 ④ Build Profile   Extract structured data from CV + supplement from cover letter:
     │              name, degree, institution, courses taught, technologies, research areas
     ▼
 ⑤ Evaluate        Rate teaching (1–5⭐), research (1–5⭐), fit (1–5⭐),
     │              identify red flags, summarize recommendation letters,
     ▼              generate overall recommendation
 ⑥ Save Results    JSON files in data/results/{applicant_name}/
     │
     ▼
 ⑦ Rank            Comparative ranking across the entire pool (auto-runs after processing)
     │
     ▼
 ⑧ Review          Committee reviews in viewer → advance through stage pipeline
```

### What happens at each stage

**Extract** — Tries pdfplumber first for PDFs, falls back to pypdf. For DOCX, prefers pandoc (better structure preservation), falls back to python-docx. Legacy `.doc` files are converted via LibreOffice if available. Text is cleaned (whitespace normalized, encoding fixed) and optionally chunked for long documents.

**Categorize** — Fast-path: filename patterns like `CV_*.pdf` or `Teaching*.docx` are matched without an LLM call. Remaining documents get a text sample (first 3000 chars) sent to the LLM for classification.

**RAG Index** — Each applicant's documents are uploaded to GenAI Studio, a per-applicant knowledge base is created, files are linked, and the system waits for server-side chunking and embedding (configurable, default 12s). Metadata is saved in `rag_metadata.json` for reuse on subsequent runs.

**Build Profile** — The CV is the primary source. The LLM extracts structured fields (degree, institution, courses, technologies, etc.) into JSON. The cover letter supplements with motivation, university-specific interest, and teaching philosophy signals.

**Evaluate** — Three evaluation steps (teaching, research, fit/flags) are run using position-aware prompts (rubrics, red flags, and emphasis injected from config or `--position` YAML). In RAG mode, prompts are shorter and reference the KB for full document retrieval. In ensemble mode, each step runs across multiple models independently, then a synthesis model merges the results. Recommendation letters are summarized individually.

**Rank** — After all applicants are processed, a comparison table is built and sent to the LLM for curve-adjusted scores and tier assignments (Top, Strong, Middle, Below Average, Weak) relative to the pool. The primary dimension (from config) is weighted more heavily.

**Review** — The committee uses the web viewer to review evaluations, override ratings, add notes, and advance applicants through the stage pipeline from Received through to Offer (or reject at any point).

---

## Evaluation Modes

The pipeline supports three combinable modes. Each is opt-in via CLI flags.

### Classic Mode (`--no-rag --no-ensemble`)

The original approach. Document text is truncated (3000–4000 chars per document) and stuffed directly into the evaluation prompt. Fast, simple, works with any model. Best for quick first passes or when GenAI Studio is under heavy load.

### RAG Mode (default, disable with `--no-rag`)

Each applicant's full documents are uploaded to GenAI Studio and indexed into a per-applicant knowledge base. Evaluation prompts use `collections=[kb_id]` to let the LLM retrieve relevant passages from the full documents, avoiding truncation artifacts.

Benefits over classic mode:
- Full document access (no truncation)
- Better evidence citation (LLM retrieves specific passages)
- Enables the post-processing chat interface
- More accurate evaluation for applicants with lengthy CVs or detailed statements

### Ensemble Mode (disabled by default, enable in `src/config.py`)

Each evaluation step (teaching, research, fit) is run independently across multiple models:

| Role | Default Models |
|------|----------------|
| Ensemble members | `llama3.3:70b`, `llama4:latest`, `qwen2.5:72b`, `gemma3:27b` |
| Synthesizer | `deepseek-r1:70b` |

The synthesizer reads all individual outputs and writes a single authoritative assessment. It weighs evidence quality rather than simply averaging ratings. Per-model breakdowns are saved in `evaluation_ensemble.json` for transparency.

Benefits:
- Reduced single-model bias
- More robust star ratings
- Richer strengths/weaknesses (union of findings across models)
- Full auditability of which model said what

### Ensemble Ranking (`rank --ensemble`)

The comparative ranking step can also use ensemble mode. The same comparison table is sent to each ensemble model independently, and the synthesizer merges the individual rankings into one consensus using median curved scores, majority-rule tiers, and re-assigned ranks. Individual model outputs are saved in `comparative_ensemble.json`.

### Combining Modes

RAG and ensemble are fully composable:

```bash
# RAG + single model (default)
python run.py process ../vap-search-2025/

# RAG + ensemble (edit config.py: ensemble.enabled = True)
python run.py process ../vap-search-2025/

# Classic + single model
python run.py process ../vap-search-2025/ --no-rag --no-ensemble

# Classic + ensemble (unusual but supported)
python run.py process ../vap-search-2025/ --no-rag

# Ensemble ranking (re-rank only, after processing)
python run.py rank --ensemble
python run.py rank --ensemble --position positions/vap-teaching-2025.yaml
```

---

## Stage Pipeline

The web viewer includes a hiring pipeline that tracks each applicant's progression through the review process. Stages are managed entirely through the viewer interface — no CLI needed after initial processing.

### Stages

```
📥 Received → 🔍 Screening → 📋 Short List → 🎤 Interview → 🏛️ Campus Visit → 🎉 Offer

                              ✕ Rejected (from any stage)
```

| Stage | Meaning |
|-------|---------|
| 📥 Received | Application processed by the pipeline, awaiting committee review |
| 🔍 Screening | Under active review (reading materials, checking qualifications) |
| 📋 Short List | Identified as a strong candidate for further consideration |
| 🎤 Interview | Selected for phone/video interview |
| 🏛️ Campus Visit | Invited for on-campus visit and job talk |
| 🎉 Offer | Position offered to this candidate |
| ✕ Rejected | Not advancing (tracks which stage they were rejected from) |

### How it works

**Dashboard** — A pipeline bar across the top shows all stages with applicant counts. Click any stage to filter the view. Each applicant card has ▶ Advance and ✕ Reject buttons directly on it for quick triage without opening detail pages.

**Applicant Detail** — A clickable stage progression bar lets you jump to any stage directly (for corrections) or use the Advance/Reject action buttons. Rejecting prompts for an optional reason. Rejected applicants can be restored to Received.

**Stage History** — Every stage transition is logged with a timestamp in `human_review.json`, providing a full audit trail of committee decisions.

### Filtering by stage

The dashboard pipeline bar doubles as a filter. Click a stage to see only applicants at that point in the process. The "All" view shows everyone with stage badges on their cards. Stage filtering combines with existing sort/filter controls (recommendation, teaching stars, etc.).

### Data storage

Stage data is stored in the existing `human_review.json` per applicant:

```json
{
  "stage": "short_list",
  "stage_history": [
    {"from": "received", "to": "screening", "at": "2025-02-18T10:30:00"},
    {"from": "screening", "to": "short_list", "at": "2025-02-19T14:15:00"}
  ],
  "rejected_from": null,
  "rejection_reason": null
}
```

Applicants without a `stage` field default to "received" — fully backward-compatible with existing results.

### Stage API

| Method | Endpoint | Body |
|--------|----------|------|
| `POST` | `/api/applicant/<n>/stage` | `{"action": "advance"}` |
| `POST` | `/api/applicant/<n>/stage` | `{"action": "reject", "reason": "..."}` |
| `POST` | `/api/applicant/<n>/stage` | `{"stage": "short_list"}` (direct set) |
| `GET` | `/api/stages/summary` | — (returns counts per stage) |

---

## CLI Reference

### Global Options

```bash
python run.py [-v|--verbose] <command> [options]
```

### `process` — Analyze Applications

```bash
python run.py process <applicants_dir> [options]
```

| Flag | Description |
|------|-------------|
| `--model MODEL` | LLM model name (default: from config) |
| `--results-dir DIR` | Override results directory |
| `--reprocess` | Re-process applicants even if results exist |
| `--skip-rank` | Skip comparative ranking after processing |
| `--no-rag` | Disable RAG (use text-stuffed prompts) |
| `--no-ensemble` | Disable ensemble (use single-model evaluation) |
| `--force-recreate-kbs` | Delete and recreate KBs even if they exist |
| `--position YAML` | Path to position profile YAML (default: VAP teaching) |

Examples:

```bash
# All applicants (default VAP teaching config)
python run.py process ../vap-search-2025/

# Single applicant
python run.py process ../vap-search-2025/Alice_Smith/

# Faster model for quick pass
python run.py process ../vap-search-2025/ --model gemma3:12b --no-rag --no-ensemble

# Force reprocess everyone
python run.py process ../vap-search-2025/ --reprocess

# Rebuild knowledge bases from scratch
python run.py process ../vap-search-2025/ --force-recreate-kbs

# Custom position profile
python run.py process ../tt-search-2026/ --position positions/tt-research-2026.yaml
```

### `rank` — Comparative Ranking

Re-run (or run for the first time) the cross-pool comparative ranking on already-processed results.

```bash
python run.py rank [--model MODEL] [--results-dir DIR] [--position YAML] [--ensemble]
```

| Flag | Description |
|------|-------------|
| `--model MODEL` | LLM model name |
| `--results-dir DIR` | Override results directory |
| `--position YAML` | Path to position profile YAML |
| `--ensemble` | Multi-model ensemble ranking (runs across all ensemble models, then synthesizes consensus) |

Examples:

```bash
# Re-rank with a different model
python run.py rank --model deepseek-r1:70b

# Ensemble consensus ranking
python run.py rank --ensemble

# Ensemble ranking with position context
python run.py rank --ensemble --position positions/vap-teaching-2025.yaml
```

### `serve` — Launch Web Viewer

```bash
python run.py serve [options]
```

| Flag | Description |
|------|-------------|
| `--port PORT` | Port number (default: 5000) |
| `--results-dir DIR` | Results directory |
| `--applicants-dir DIR` | Applicants directory (enables document viewing) |
| `--model MODEL` | LLM model for chat (default: evaluation model from config) |
| `--no-chat` | Disable chat interface (viewer-only, no LLM connection needed) |
| `--position YAML` | Path to position profile YAML (provides context for chat system prompt) |

Examples:

```bash
# Basic viewer
python run.py serve --port 8080

# Full viewer with chat and position context
python run.py serve --applicants-dir ../vap-search-2025/ --position positions/vap-teaching-2025.yaml
```

### `status` — Check Processing Status

```bash
python run.py status <applicants_dir> [--results-dir DIR]
```

Shows a table with document counts, processing status, RAG status, and ensemble status for each applicant.

### `export` — Export to CSV

```bash
python run.py export <applicants_dir> [--output FILE] [--results-dir DIR]
```

Exports all processed results (including human review overrides, stage pipeline status, comparative rankings, and processing metadata) to a CSV file. Default output: `applicant_summary.csv`. Stage columns include: Stage, Rejected From, Rejection Reason.

### `cleanup-kbs` — Delete Knowledge Bases

```bash
python run.py cleanup-kbs [--results-dir DIR]
```

Interactively deletes all RAG knowledge bases from the GenAI Studio server. Requires typing `DELETE ALL KBS` to confirm. For a safer experience with full confirmation prompts, use `cleanup_rag.py` directly.

---

## Web Viewer

The viewer runs locally at `http://127.0.0.1:5000` and provides two main views.

### Dashboard

The dashboard shows a **stage pipeline bar** across the top with clickable stages and applicant counts — click any stage to filter the view. Below is a sortable grid showing all applicants with star ratings, tier badges, recommendation summaries, stage badges, and review status. Each card includes quick-action ▶ Advance and ✕ Reject buttons for fast triage.

Sort options: teaching stars, research stars, fit score, comparative rank, name. Filter options: recommendation level, minimum teaching stars, stage, reviewed/unreviewed. A live search box filters by name.

### Applicant Detail Page

A comprehensive single-page view for each applicant:

- **Profile card** — Name, degree, institution, current position, adjacent field badge
- **Stage pipeline** — Clickable progression bar showing current stage, advance/reject buttons with full stage history tracking
- **Evaluation** — Teaching/research/fit ratings with detailed justifications
- **Red flags** — Severity-tagged concerns with explanations
- **Recommendation letters** — Individual summaries with tone and key quotes
- **Executive summary** — Overall assessment
- **Ensemble breakdown** — Per-model results (collapsible, if ensemble was used)
- **Comparative ranking** — Curved scores and tier assignment relative to pool
- **Human review panel** — Override any LLM-generated field, add committee comments
- **Chat panel** — Floating chat interface to ask follow-up questions against the applicant's RAG knowledge base
- **Processing metadata** — Collapsible section showing which models and modes were used

### Chat Interface

When RAG knowledge bases are available and the viewer is launched with an LLM client, a floating chat panel appears on each applicant's detail page. It queries the applicant's KB with a position-aware system prompt (automatically using `--position` config if provided, or reading from `processing_meta.json`) and includes quick-ask buttons for common questions:

- "Summarize their teaching experience"
- "What statistics courses have they taught?"
- "What do their recommendation letters say?"
- "Describe their research agenda"

---

## Human Review Overlay

The viewer supports a human review system that stores committee corrections, stage progression, and overrides in a separate `human_review.json` file without modifying the original LLM outputs. The viewer merges these on-the-fly when rendering.

### Overridable fields

- Teaching, research, and fit ratings (star widgets)
- Overall recommendation (Strong / Moderate / Weak / Do Not Advance)
- Profile fields (name, degree, institution, current position, etc.)
- Boolean toggles (statistics-adjacent, AI in education)
- Editable lists (technologies, research areas)
- Committee comments, teaching notes, research notes

### Stage tracking

- Current stage in the hiring pipeline
- Full stage history with timestamps
- Rejection stage and optional rejection reason

### Design principle

LLM outputs are **immutable** — human corrections are stored separately and merged at display time. This preserves the full audit trail: you can always see what the LLM originally said versus what the committee changed.

---

## Configuration

All settings live in `src/config.py` as Python dataclasses. Edit directly to customize, or use `--position` YAML files for position-specific overrides.

### Key Configuration Sections

| Dataclass | What it controls |
|-----------|-----------------|
| `LLMConfig` | Model names for classification/extraction/evaluation, temperatures, token limits |
| `EnsembleConfig` | Ensemble member models, synthesizer model, temperatures, toggle |
| `RAGConfig` | Enable/disable RAG, indexing wait times, KB retention |
| `PositionConfig` | Position title, department, description, adjacent field list |
| `PositionWeights` | Primary/secondary dimension, weight multipliers for combined scoring |
| `RatingRubric` | Star rating criteria for teaching (1–5) and research (1–5) |
| `RedFlagCriteria` | What red flags to check, severity levels |
| `ViewerConfig` | Host, port, debug mode |
| `Config` | Master config combining all of the above |

### Common Customizations

**Change the default LLM model:**
```python
# src/config.py
class LLMConfig:
    classification_model: str = "gemma3:12b"     # faster for categorization
    extraction_model: str = "gemma3:12b"          # faster for profile building
    evaluation_model: str = "llama3.3:70b"        # stronger for evaluation
```

**Enable ensemble by default:**
```python
class EnsembleConfig:
    enabled: bool = True
```

**Adjust RAG indexing wait time:**
```python
class RAGConfig:
    index_wait_seconds: int = 20  # increase for large documents
```

**Use a position profile (preferred over editing config.py):**
```bash
python run.py process ../search/ --position positions/tt-research-2026.yaml
```

**Customize position in config.py (alternative to YAML):**
```python
class PositionConfig:
    position_title: str = "Assistant Professor"
    position_focus: str = "Research (Primary)"
    position_duration: str = "Tenure-track"
```

---

## Output Files

For each processed applicant, the following JSON files are saved in `data/results/{folder_name}/`:

| File | Contents |
|------|----------|
| `profile.json` | Structured applicant profile (name, degree, courses, technologies, etc.) |
| `evaluation.json` | Final evaluation (synthesized if ensemble was used) with star ratings, justifications, red flags, letter summaries, executive summary |
| `evaluation_ensemble.json` | Per-model breakdowns from each ensemble member (if ensemble enabled) |
| `documents.json` | Document metadata — filenames, categories, page counts, relative source path (no raw text stored) |
| `processing_meta.json` | Processing flags — RAG/ensemble enabled, models used, position metadata, timestamp |
| `rag_metadata.json` | KB ID, file IDs, creation timestamp (if RAG enabled) |
| `comparative.json` | Curved scores, tier, rank (written after comparative ranking) |
| `comparative_ensemble.json` | Per-model ranking breakdowns (if ensemble ranking used) |
| `human_review.json` | Committee overrides, comments, stage pipeline status, stage history (written by viewer) |

Raw document text is **never** stored on disk — only metadata. Text stays in memory during processing for privacy.

---

## RAG Knowledge Base Cleanup

After the hiring process is complete, clean up knowledge bases and uploaded files from the GenAI Studio server.

### Interactive cleanup (recommended)

```bash
python cleanup_rag.py
```

This has three layers of safety confirmation:
1. "Is the hiring process fully complete?" (yes/no)
2. Type `DELETE ALL KBS` to confirm
3. 3-second countdown before execution

### Other options

```bash
# Preview what would be deleted
python cleanup_rag.py --dry-run

# Skip confirmations (for scripting — DANGEROUS)
python cleanup_rag.py --yes

# Keep uploaded files, only delete KBs
python cleanup_rag.py --keep-files

# Quick cleanup via run.py
python run.py cleanup-kbs
```

After cleanup, the chat interface will no longer work. Local results (profiles, evaluations, reviews) are **not** deleted.

---

## Evaluation Criteria

The rubrics and red flags below are the **defaults** for the VAP teaching position. When using a `--position` YAML profile, these are replaced by the values defined in that profile.

### Teaching (Primary Focus — default)

| Stars | Criteria |
|-------|----------|
| ⭐⭐⭐⭐⭐ | 5+ courses as instructor, excellent evaluations, innovative pedagogy, AI/tech integration, statistics-specific courses, evidence of course development |
| ⭐⭐⭐⭐ | 3–5 courses as instructor, clear teaching philosophy, some technology integration, positive evaluations |
| ⭐⭐⭐ | 1–3 courses or significant TA experience, standard teaching statement, basic technology use |
| ⭐⭐ | TA only, generic teaching statement, no evidence of independent instruction |
| ⭐ | No teaching experience, no teaching statement, or major concerns about teaching ability |

### Research (Secondary — default)

| Stars | Criteria |
|-------|----------|
| ⭐⭐⭐⭐⭐ | Active research agenda, publications in statistics journals, potential for collaboration, funded research |
| ⭐⭐⭐⭐ | Some publications, clear research direction, statistics-relevant area |
| ⭐⭐⭐ | Dissertation research with reasonable potential, some conference presentations |
| ⭐⭐ | Minimal research activity, few or no publications |
| ⭐ | No research or non-statistics research with no clear connection |

### Red Flags Checked (default)

- Degree not in Statistics or an adjacent field
- ABD status with no defense date or timeline
- Employment gaps > 1 year without explanation
- No teaching experience whatsoever
- Generic/boilerplate cover letter (not tailored to the university)
- Letters of recommendation that contradict CV claims
- Primarily research-focused with no teaching interest expressed
- Concerns raised in recommendation letters
- Mismatched dates or inconsistencies across documents
- No mention of statistics-relevant coursework or training

Each red flag is assigned a severity: Minor, Moderate, or Serious.

### Comparative Ranking

After individual evaluations, a cross-pool comparison assigns:
- **Curve-adjusted scores** — Teaching and research stars re-calibrated relative to the pool
- **Tier** — Top, Strong, Middle, Below Average, Weak
- **Rank** — Numeric position ordered by the primary dimension (from config) then secondary
- **Comparative notes** — LLM-generated notes explaining the ranking rationale

For large pools (>20 applicants), ranking is done in batches with a merge pass. With `--ensemble`, multiple models rank the pool independently and a synthesizer produces the consensus ranking.

---

## Dependencies

### Required

| Package | Purpose |
|---------|---------|
| `openai` | GenAI Studio API client (used by genai_studio.py) |
| `httpx` | HTTP client for RAG/model endpoints |
| `PyYAML` | Position profile YAML loading |
| `click` | CLI command framework |
| `rich` | Progress bars, tables, console output |
| `flask` | Web viewer |
| `pdfplumber` | Primary PDF text extraction |
| `pypdf` | Fallback PDF extraction |
| `python-docx` | DOCX text extraction |

### Optional

| Package | Purpose |
|---------|---------|
| `pandoc` (system) | Better DOCX structure preservation |
| `LibreOffice` (system) | Legacy `.doc` file conversion |
| `numpy` | Faster cosine similarity in genai_studio.py |

### Installation

```bash
pip install -r requirements.txt
```

### GenAI Studio SDK

The `genai_studio.py` file is not included in this repository — it's a separate Purdue-specific module. Copy the version with RAG support (v1.2+) into the project root. Verify with:

```bash
python -c "from genai_studio import GenAIStudio, FileInfo, KnowledgeBase, RAGError; print('✅ RAG imports OK')"
```

---

## Troubleshooting

### 504 Gateway Timeout during processing

The server's reverse proxy kills slow requests after ~5 minutes. This typically happens with 70B models during categorization or profile extraction. Solutions:
- Use a smaller/faster model for categorization: edit `classification_model` in `src/config.py`
- Use `--model gemma3:12b` for the initial pass
- Retry — cold-start timeouts often resolve on second attempt

### `'GenAIStudio' object has no attribute 'upload_file'`

You're using an older version of `genai_studio.py` that doesn't have RAG methods. Copy the newer version (v1.2+) with `upload_file`, `create_knowledge_base`, etc. Or run with `--no-rag` to skip RAG entirely.

### `ModuleNotFoundError: No module named 'yaml'`

Install PyYAML: `pip install PyYAML` (or `pip install -r requirements.txt`). This is needed for `--position` YAML profile loading.

### RAG queries return generic (non-grounded) responses

Indexing takes time after linking files to a knowledge base. Increase `index_wait_seconds` in `RAGConfig` (default: 12s, try 20–30s for large PDFs).

### Applicant already processed — skipping

By default, `process` skips applicants that already have an `evaluation.json`. Use `--reprocess` to force re-evaluation, or delete the applicant's results folder.

### Chat interface disabled in viewer

The chat interface requires RAG knowledge bases to exist on the server and an LLM client connection. Make sure:
- You processed with RAG enabled (no `--no-rag`)
- `rag_metadata.json` exists for the applicant
- The viewer was launched without `--no-chat`
- `GENAI_STUDIO_API_KEY` is set

### Document links broken after moving to another machine

Document paths in `documents.json` may be absolute from the original machine. Fix with:

```bash
python fix_paths.py ./vap-search-2025/ --dry-run   # preview
python fix_paths.py ./vap-search-2025/              # apply
```

Or always pass `--applicants-dir` when serving (bypasses stored paths entirely):

```bash
python run.py serve --applicants-dir ./vap-search-2025/
```

### Empty text extraction from PDFs

Some PDFs (especially scanned documents) yield no text. The pipeline reports this as "No text could be extracted." Consider OCR preprocessing or skipping that applicant.

---

## Notes

- The viewer runs **locally only** (127.0.0.1) — no external network access
- Results are stored as JSON in `data/results/{applicant_name}/`
- Raw document text is **not** stored on disk (only metadata) for privacy
- Re-running `process` on an applicant will skip unless `--reprocess` is set
- The tool works with any model available on Purdue GenAI Studio
- Human review data is stored separately and never overwrites LLM outputs
- Stage pipeline is fully backward-compatible — existing results default to "received"
- Document paths are stored as relative paths for portability across machines
- The `cleanup_rag.py` script has 3-layer safety confirmation for irreversible operations
- Without `--position`, all behavior matches the original VAP teaching defaults

---

## License

Internal use for Purdue University Department of Statistics hiring processes.

## Author

Timothy Reese — Department of Statistics, Purdue University