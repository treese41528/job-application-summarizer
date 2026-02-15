"""
Comparative Ranker — post-processing stage that re-scores all candidates
relative to the applicant pool.

Individual evaluations rate each applicant in isolation (absolute).
This module loads all evaluated profiles, builds a compact comparison
table, and asks the LLM to assign curve-adjusted scores so that the
committee can see who stands out *within this specific pool*.

Produces a `comparative.json` in the results root with:
  - teaching_rank / research_rank (1 = best)
  - teaching_curved / research_curved (1-5 on a curve)
  - tier: "Top", "Strong", "Middle", "Below Average", "Weak"
  - comparative_notes: brief LLM justification

Does NOT overwrite individual evaluation.json files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.llm.client import LLMClient
from src.config import Config, EnsembleConfig

logger = logging.getLogger(__name__)

# Maximum candidates per LLM call (context window budget)
BATCH_SIZE = 40


def run_comparative_ranking(
    results_dir: Path,
    client: LLMClient,
    config: Optional[Config] = None,
    ensemble_config: Optional[EnsembleConfig] = None,
) -> dict[str, Any]:
    """
    Load all evaluated applicants, rank them comparatively, and save results.

    Args:
        results_dir: Directory containing per-applicant result folders.
        client: LLM client for comparative prompts.
        config: Position configuration. When provided, prompts and sort
                order respect the configured primary/secondary weights.
        ensemble_config: If provided and enabled, runs ranking across
                multiple models and synthesizes a consensus ranking.

    Returns the full comparative results dict.
    """
    use_ensemble = ensemble_config is not None and ensemble_config.enabled

    # 1. Load all evaluated applicants
    candidates = _load_all_candidates(results_dir)

    if len(candidates) < 2:
        logger.warning("Need at least 2 candidates for comparative ranking")
        print("  ⚠️  Need at least 2 evaluated candidates. Skipping comparative ranking.")
        return {}

    mode = "ensemble" if use_ensemble else "single model"
    print(f"  📊 Comparing {len(candidates)} candidates ({mode})...")

    # 2. Build comparison table
    table = _build_comparison_table(candidates)

    # 3. Get LLM comparative rankings (batch if needed)
    if len(candidates) <= BATCH_SIZE:
        if use_ensemble:
            rankings = _rank_batch_ensemble(table, client, config, ensemble_config)
        else:
            rankings = _rank_batch(table, client, config)
    else:
        rankings = _rank_large_pool(table, candidates, client, config)

    # 3. Get LLM comparative rankings (batch if needed)
    ensemble_individuals = None
    if len(candidates) <= BATCH_SIZE:
        if use_ensemble:
            rankings, ensemble_individuals = _rank_batch_ensemble(
                table, client, config, ensemble_config,
            )
        else:
            rankings = _rank_batch(table, client, config)
    else:
        rankings = _rank_large_pool(table, candidates, client, config)

    if not rankings:
        print("  ❌ Comparative ranking failed (LLM returned no results). Try a faster model or re-run.")
        return {}

    # 4. Merge rankings back with candidate data
    result = _build_final_result(candidates, rankings, config)

    if use_ensemble:
        result["ranking_mode"] = "ensemble"
        result["ensemble_models"] = ensemble_config.models
        result["synthesizer_model"] = ensemble_config.synthesizer_model

    # 5. Save
    output_file = results_dir / "comparative.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Comparative rankings saved to {output_file}")

    # Save ensemble individual model outputs (if any)
    if ensemble_individuals:
        ensemble_file = results_dir / "comparative_ensemble.json"
        with open(ensemble_file, "w") as f:
            json.dump(ensemble_individuals, f, indent=2)
        logger.info(f"Ensemble ranking details saved to {ensemble_file}")

    # 6. Also save per-applicant comparative data
    for entry in result.get("rankings", []):
        folder = entry.get("folder_name")
        if folder:
            per_applicant_file = results_dir / folder / "comparative.json"
            try:
                with open(per_applicant_file, "w") as f:
                    json.dump(entry, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save comparative for {folder}: {e}")

    return result


def _load_all_candidates(results_dir: Path) -> list[dict]:
    """Load profile + evaluation summaries for all processed applicants."""
    candidates = []

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue

        profile_file = folder / "profile.json"
        eval_file = folder / "evaluation.json"

        if not profile_file.exists() or not eval_file.exists():
            continue

        try:
            with open(profile_file) as f:
                profile = json.load(f)
            with open(eval_file) as f:
                evaluation = json.load(f)

            candidates.append({
                "folder_name": folder.name,
                "profile": profile,
                "evaluation": evaluation,
            })
        except Exception as e:
            logger.warning(f"Skipping {folder.name} in comparative: {e}")

    return candidates


def _build_comparison_table(candidates: list[dict]) -> str:
    """
    Build a compact text table summarizing all candidates for the LLM.
    Includes key differentiators without overwhelming the context window.
    """
    lines = []
    lines.append("CANDIDATE COMPARISON TABLE")
    lines.append("=" * 80)

    for i, c in enumerate(candidates, 1):
        p = c["profile"]
        e = c["evaluation"]

        # Count instructor-level courses
        instructor_courses = sum(
            ct.get("semesters", 1)
            for ct in p.get("courses_taught", [])
            if ct.get("role") in ("instructor", "co-instructor")
        )
        ta_courses = sum(
            1 for ct in p.get("courses_taught", [])
            if ct.get("role") == "ta"
        )

        lines.append(f"\n--- Candidate {i}: {p.get('name', 'Unknown')} ---")
        lines.append(f"  Folder: {c['folder_name']}")
        lines.append(f"  Degree: {p.get('terminal_degree', '?')} from {p.get('degree_institution', '?')}")
        lines.append(f"  Degree Status: {p.get('degree_status', '?')}")
        lines.append(f"  Stats-Adjacent: {'Yes' if p.get('is_statistics_adjacent') else 'No'}")
        lines.append(f"  Current Role: {p.get('current_position', '?')} at {p.get('current_institution', '?')}")
        lines.append(f"  Teaching: {instructor_courses} courses as instructor, {ta_courses} as TA")
        lines.append(f"  Teaching Techs: {', '.join(p.get('teaching_technologies', [])) or 'None listed'}")
        lines.append(f"  AI in Education: {'Yes' if p.get('ai_in_education') else 'No'}")
        lines.append(f"  Research Areas: {', '.join(p.get('research_areas', [])[:5]) or 'None listed'}")
        lines.append(f"  Publications: {p.get('publications_count', 0)}")

        # Individual eval summary
        lines.append(f"  -- Individual Eval --")
        lines.append(f"  Teaching Stars: {e.get('teaching_stars', 0)}/5")
        lines.append(f"  Research Stars: {e.get('research_stars', 0)}/5")
        lines.append(f"  Fit Score: {e.get('fit_score', 0)}/5")
        lines.append(f"  Recommendation: {e.get('overall_recommendation', '?')}")
        lines.append(f"  Red Flags: {len(e.get('red_flags', []))}")

        # Teaching strengths (brief)
        strengths = e.get("teaching_strengths", [])[:2]
        if strengths:
            lines.append(f"  Teaching Strengths: {'; '.join(strengths)}")
        weaknesses = e.get("teaching_weaknesses", [])[:2]
        if weaknesses:
            lines.append(f"  Teaching Gaps: {'; '.join(weaknesses)}")

        # Letters consensus
        if e.get("letter_consensus"):
            lines.append(f"  Letters: {e['letter_consensus'][:150]}")

    return "\n".join(lines)


def _rank_batch(
    table: str,
    client: LLMClient,
    config: Optional[Config] = None,
) -> list[dict]:
    """Send the comparison table to the LLM and get comparative rankings."""

    # Derive position context from config or use defaults
    if config:
        position_summary = config.position_summary()
        department = config.position.department
        university = config.position.university
        primary = config.weights.primary        # "teaching" or "research"
        secondary = config.weights.secondary
    else:
        position_summary = "Visiting Assistant Professor (Teaching Focus, 2-year term)"
        department = "the Department of Statistics"
        university = "Purdue University"
        primary = "teaching"
        secondary = "research"

    prompt = f"""You are a faculty search committee assistant for a {position_summary}
position in {department} at {university}. This is a {primary.upper()}-FOCUSED position.

Below is a summary of all candidates who have been individually evaluated. Your job is to
COMPARATIVELY rank them against each other — not in isolation, but relative to this specific
applicant pool.

{table}

For EACH candidate, provide:
1. {primary}_rank: Integer rank for {primary} (1 = strongest in the pool)
2. {secondary}_rank: Integer rank for {secondary} (1 = strongest in the pool)
3. {primary}_curved: A 1-5 star rating ON A CURVE relative to this pool
   - 5 = top ~10% of THIS pool
   - 4 = above average in THIS pool
   - 3 = average for THIS pool
   - 2 = below average in THIS pool
   - 1 = weakest in THIS pool
4. {secondary}_curved: Same curve logic for {secondary}
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
- Weight {primary} MORE than {secondary} (this is a {primary}-focused position)
- Consider degree field fit for the department
- A candidate with 4/5 individual {primary} but less experience than others in the pool
  might get curved to 3/5

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
    "comparative_notes": "Strongest {primary} profile in the pool with..."
  }},
  ...
]"""

    try:
        rankings = client.query_json(prompt, temperature=0.1, max_tokens=4096)

        if isinstance(rankings, dict) and "rankings" in rankings:
            rankings = rankings["rankings"]

        if not isinstance(rankings, list):
            logger.error(f"Expected list from comparative ranking, got {type(rankings)}")
            return []

        return rankings

    except Exception as e:
        logger.error(f"Comparative ranking failed: {e}")
        return []


def _rank_large_pool(
    table: str,
    candidates: list[dict],
    client: LLMClient,
    config: Optional[Config] = None,
) -> list[dict]:
    """
    For pools larger than BATCH_SIZE, split into batches, rank within each,
    then do a final merge pass on the top candidates.
    """
    # Determine weights for combined sort
    if config:
        pw = config.weights.primary_weight
        sw = config.weights.secondary_weight
        primary = config.weights.primary
    else:
        pw, sw = 2.0, 1.0
        primary = "teaching"

    # Split candidates into batches
    batches = []
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        batches.append(batch)

    print(f"  📊 Large pool ({len(candidates)} candidates) — ranking in {len(batches)} batches...")

    all_rankings = []

    for batch_idx, batch in enumerate(batches):
        print(f"    Batch {batch_idx + 1}/{len(batches)} ({len(batch)} candidates)...")
        batch_table = _build_comparison_table(batch)
        batch_rankings = _rank_batch(batch_table, client, config)
        all_rankings.extend(batch_rankings)

    # Normalize ranks across batches using weighted curved scores
    if primary == "teaching":
        all_rankings.sort(
            key=lambda r: -(r.get("teaching_curved", 0) * pw + r.get("research_curved", 0) * sw)
        )
    else:
        all_rankings.sort(
            key=lambda r: -(r.get("research_curved", 0) * pw + r.get("teaching_curved", 0) * sw)
        )

    # Re-assign global ranks
    for i, r in enumerate(all_rankings, 1):
        r["teaching_rank"] = i  # Approximate — full re-rank would need another LLM call

    all_rankings.sort(key=lambda r: -r.get("research_curved", 0))
    for i, r in enumerate(all_rankings, 1):
        r["research_rank"] = i

    return all_rankings


def _build_final_result(
    candidates: list[dict],
    rankings: list[dict],
    config: Optional[Config] = None,
) -> dict[str, Any]:
    """Merge rankings with candidate metadata into final result."""

    primary = config.weights.primary if config else "teaching"

    # Index rankings by folder_name
    rank_map = {}
    for r in rankings:
        folder = r.get("folder_name", "")
        rank_map[folder] = r

    merged = []
    for c in candidates:
        folder = c["folder_name"]
        r = rank_map.get(folder, {})

        merged.append({
            "folder_name": folder,
            "name": c["profile"].get("name", "Unknown"),
            "teaching_rank": r.get("teaching_rank", 0),
            "research_rank": r.get("research_rank", 0),
            "teaching_curved": r.get("teaching_curved", 0),
            "research_curved": r.get("research_curved", 0),
            "tier": r.get("tier", "Unranked"),
            "comparative_notes": r.get("comparative_notes", ""),
            # Include original scores for reference
            "teaching_individual": c["evaluation"].get("teaching_stars", 0),
            "research_individual": c["evaluation"].get("research_stars", 0),
            "recommendation_individual": c["evaluation"].get("overall_recommendation", ""),
        })

    # Sort by tier priority then primary dimension rank
    tier_order = {"Top": 0, "Strong": 1, "Middle": 2, "Below Average": 3, "Weak": 4, "Unranked": 5}
    primary_rank_key = f"{primary}_rank"
    merged.sort(key=lambda x: (tier_order.get(x["tier"], 5), x.get(primary_rank_key, 0)))

    # Pool stats
    n = len(merged)
    avg_teaching = sum(m["teaching_individual"] for m in merged) / n if n else 0
    avg_research = sum(m["research_individual"] for m in merged) / n if n else 0

    result = {
        "pool_size": n,
        "primary_dimension": primary,
        "pool_stats": {
            "avg_teaching_individual": round(avg_teaching, 2),
            "avg_research_individual": round(avg_research, 2),
            "top_tier_count": sum(1 for m in merged if m["tier"] == "Top"),
            "strong_tier_count": sum(1 for m in merged if m["tier"] == "Strong"),
        },
        "rankings": merged,
    }

    # Include position info if config provided
    if config:
        result["position"] = config.position_summary()

    return result


# ════════════════════════════════════════════════════════════════════════════
# ENSEMBLE RANKING
# ════════════════════════════════════════════════════════════════════════════


def _rank_batch_ensemble(
    table: str,
    client: LLMClient,
    config: Optional[Config],
    ensemble_config: EnsembleConfig,
) -> tuple[list[dict], list[dict]]:
    """
    Run comparative ranking across multiple models, then synthesize.

    Same prompt goes to each ensemble model independently.  A synthesizer
    model then merges the individual rankings into one consensus ranking.

    Returns:
        (synthesized_rankings, individual_model_results)
    """
    models = ensemble_config.models
    logger.info(f"  Ensemble ranking across {len(models)} models…")
    print(f"    🔄 Running ranking on {len(models)} models: {', '.join(models)}")

    # Build the ranking prompt (same one used by single-model _rank_batch)
    if config:
        position_summary = config.position_summary()
        department = config.position.department
        university = config.position.university
        primary = config.weights.primary
        secondary = config.weights.secondary
    else:
        position_summary = "Visiting Assistant Professor (Teaching Focus, 2-year term)"
        department = "the Department of Statistics"
        university = "Purdue University"
        primary = "teaching"
        secondary = "research"

    prompt = f"""You are a faculty search committee assistant for a {position_summary}
position in {department} at {university}. This is a {primary.upper()}-FOCUSED position.

Below is a summary of all candidates who have been individually evaluated. Your job is to
COMPARATIVELY rank them against each other — not in isolation, but relative to this specific
applicant pool.

{table}

For EACH candidate, provide:
1. teaching_rank: Integer rank for teaching (1 = strongest teacher in the pool)
2. research_rank: Integer rank for research (1 = strongest researcher in the pool)
3. teaching_curved: A 1-5 star rating ON A CURVE relative to this pool
   - 5 = top ~10%, 4 = above average, 3 = average, 2 = below average, 1 = weakest
4. research_curved: Same curve logic for research
5. tier: One of "Top", "Strong", "Middle", "Below Average", "Weak"
6. comparative_notes: 1-2 sentences on what differentiates this candidate.

IMPORTANT:
- Rankings must be unique integers (no ties for rank)
- Curved scores CAN have ties
- Weight {primary} MORE than {secondary}
- Consider degree field fit for the department

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
    "comparative_notes": "..."
  }},
  ...
]"""

    # ── Step 1: Query each model ──
    individual_results = client.ensemble_query_json(
        prompt,
        models=models,
        temperature=ensemble_config.member_temperature,
    )

    successes = []
    for r in individual_results:
        data = r.get("data")
        # Handle both list and dict-wrapped-list responses
        if isinstance(data, dict) and "rankings" in data:
            data = data["rankings"]
        if isinstance(data, list) and len(data) > 0:
            r["data"] = data  # normalise
            successes.append(r)
            print(f"      ✅ {r['model']}: ranked {len(data)} candidates")
        else:
            err = r.get("error", "empty/invalid response")
            print(f"      ❌ {r['model']}: {err}")

    logger.info(f"  Ensemble ranking: {len(successes)}/{len(models)} succeeded")

    if not successes:
        logger.error("All ensemble ranking models failed")
        return [], individual_results

    # Short-circuit: only one success → use it directly
    if len(successes) == 1:
        logger.info("  Single model succeeded — skipping synthesis")
        return successes[0]["data"], individual_results

    # ── Step 2: Format for synthesis ──
    import json as _json
    synth_lines = []
    for i, r in enumerate(successes, 1):
        synth_lines.append(f"--- Ranker {i} (Model: {r['model']}) ---")
        # Only include key fields to keep prompt manageable
        compact = []
        for entry in r["data"]:
            compact.append({
                "folder_name": entry.get("folder_name", ""),
                "name": entry.get("name", ""),
                "teaching_rank": entry.get("teaching_rank", 0),
                "research_rank": entry.get("research_rank", 0),
                "teaching_curved": entry.get("teaching_curved", 0),
                "research_curved": entry.get("research_curved", 0),
                "tier": entry.get("tier", ""),
            })
        synth_lines.append(_json.dumps(compact, indent=2))
        synth_lines.append("")

    evaluations_text = "\n".join(synth_lines)

    synthesis_prompt = f"""You are a senior faculty search committee chair synthesizing multiple independent comparative rankings of the same applicant pool.

{len(successes)} different rankers independently ranked the candidates for a {position_summary} position in {department} at {university}. {primary.capitalize()} is the primary criterion.

INDIVIDUAL RANKINGS:
{evaluations_text}

YOUR TASK:
Synthesize into ONE final consensus ranking:
1. For each candidate, determine the most justified tier by majority agreement.
2. If rankers disagree on tier, lean toward the majority or the more evidence-backed assessment.
3. For curved scores, use the median of the individual scores (round to nearest integer).
4. For ranks, re-assign unique integer ranks based on the consensus curved scores
   (primary curved × {config.weights.primary_weight if config else 2.0} + secondary curved × {config.weights.secondary_weight if config else 1.0}).
5. Write comparative_notes that reflect the consensus view. If rankers disagreed on a candidate, note that.

Respond with ONLY a JSON array, one object per candidate:
[
  {{
    "folder_name": "exact folder name",
    "name": "candidate name",
    "teaching_rank": 1,
    "research_rank": 3,
    "teaching_curved": 5,
    "research_curved": 3,
    "tier": "Top",
    "comparative_notes": "Consensus: ..."
  }},
  ...
]"""

    # ── Step 3: Synthesize ──
    print(f"    🔄 Synthesizing with {ensemble_config.synthesizer_model}...")
    try:
        synthesized = client.synthesize(
            synthesis_prompt,
            synthesizer_model=ensemble_config.synthesizer_model,
            temperature=ensemble_config.synthesizer_temperature,
            max_tokens=ensemble_config.synthesizer_max_tokens,
        )

        # Handle both list and dict-wrapped-list
        if isinstance(synthesized, dict) and "rankings" in synthesized:
            synthesized = synthesized["rankings"]
        if not isinstance(synthesized, list):
            logger.error(f"Synthesis returned {type(synthesized)}, expected list")
            # Fall back to first successful ranker
            return successes[0]["data"], individual_results

        print(f"    ✅ Synthesized consensus ranking for {len(synthesized)} candidates")
        return synthesized, individual_results

    except Exception as e:
        logger.error(f"Ranking synthesis failed: {e}")
        print(f"    ⚠️  Synthesis failed ({e}), using first ranker's output")
        return successes[0]["data"], individual_results