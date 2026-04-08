"""
Deterministic graders for all three Clinical Triage Agent tasks.

Each grader takes an episode log and returns a score in [0.0, 1.0].
Graders are pure functions: same log always produces the same score.
No randomness is used in scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import EpisodeLogEntry, TriageLevel


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _triage_entries(log: List[EpisodeLogEntry]) -> List[EpisodeLogEntry]:
    """Filter log to only TRIAGE entries."""
    return [e for e in log if e.action_type == "TRIAGE"]


def _escalate_entries(log: List[EpisodeLogEntry]) -> List[EpisodeLogEntry]:
    """Filter log to only ESCALATE entries."""
    return [e for e in log if e.action_type == "ESCALATE"]


def _ask_entries(log: List[EpisodeLogEntry]) -> List[EpisodeLogEntry]:
    """Filter log to only ASK entries."""
    return [e for e in log if e.action_type == "ASK"]


# ---------------------------------------------------------------------------
# Task 1 Grader – Easy (3 patients)
# ---------------------------------------------------------------------------

def grade_task1(
    episode_log: List[EpisodeLogEntry],
    expected_patients: int = 3,
) -> float:
    """
    Grade a Task 1 (easy) episode.

    Scoring:
    - 70%: Triage accuracy (weighted avg of level_accuracy, pathway_accuracy)
    - 20%: Speed (bonus if avg steps per patient ≤ 3)
    - 10%: Completion (all patients triaged)

    Returns: score ∈ [0.0, 1.0]
    """
    if not episode_log:
        return 0.0

    triage_entries = _triage_entries(episode_log)
    n_triaged = len(triage_entries)

    if n_triaged == 0:
        return 0.0

    # Accuracy component
    total_level_acc = sum(e.level_accuracy or 0.0 for e in triage_entries)
    total_path_acc = sum(e.pathway_accuracy or 0.0 for e in triage_entries)
    avg_level_acc = total_level_acc / n_triaged
    avg_path_acc = total_path_acc / n_triaged

    accuracy_score = 0.60 * avg_level_acc + 0.40 * avg_path_acc

    # Speed component: steps per patient
    if n_triaged > 0 and triage_entries:
        last_step = max(e.step for e in triage_entries)
        avg_steps_per_patient = last_step / n_triaged
        speed_score = 1.0 if avg_steps_per_patient <= 3 else max(0.0, 1.0 - (avg_steps_per_patient - 3) / 7)
    else:
        speed_score = 0.0

    # Completion
    completion = min(1.0, n_triaged / expected_patients)

    final_score = 0.70 * accuracy_score + 0.20 * speed_score + 0.10 * completion
    return round(max(0.0, min(1.0, final_score)), 4)


# ---------------------------------------------------------------------------
# Task 2 Grader – Medium (8 patients, limited ICU)
# ---------------------------------------------------------------------------

def grade_task2(
    episode_log: List[EpisodeLogEntry],
    expected_patients: int = 8,
    resus_capacity: int = 2,
) -> float:
    """
    Grade a Task 2 (medium) episode.

    Scoring:
    - 60%: Triage accuracy (weighted avg of level_accuracy, pathway_accuracy)
    - 15%: Information gathering (appropriate ASK usage without over-asking)
    - 15%: Resource adherence (no capacity violations)
    - 10%: Completion

    Returns: score ∈ [0.0, 1.0]
    """
    if not episode_log:
        return 0.0

    triage_entries = _triage_entries(episode_log)
    ask_entries = _ask_entries(episode_log)
    n_triaged = len(triage_entries)

    if n_triaged == 0:
        return 0.0

    # Accuracy component
    avg_level_acc = sum(e.level_accuracy or 0.0 for e in triage_entries) / n_triaged
    avg_path_acc = sum(e.pathway_accuracy or 0.0 for e in triage_entries) / n_triaged
    accuracy_score = 0.60 * avg_level_acc + 0.40 * avg_path_acc

    # Information gathering: reward useful ASKs, penalise excessive
    # Ideal: 1-2 ASK actions per patient (8-16 total for 8 patients)
    ideal_asks = expected_patients * 1.5
    n_asks = len(ask_entries)
    if n_asks == 0:
        ask_score = 0.3  # didn't bother asking; penalise
    elif n_asks <= ideal_asks:
        ask_score = min(1.0, n_asks / ideal_asks)
    else:
        # Over-asking penalty
        ask_score = max(0.0, 1.0 - (n_asks - ideal_asks) / ideal_asks)

    # Resource adherence: fraction of triage entries without penalty
    resource_violations = sum(1 for e in triage_entries if e.penalty > 0.05)
    resource_score = max(0.0, 1.0 - resource_violations / max(1, n_triaged))

    # Completion
    completion = min(1.0, n_triaged / expected_patients)

    final_score = (
        0.60 * accuracy_score
        + 0.15 * ask_score
        + 0.15 * resource_score
        + 0.10 * completion
    )
    return round(max(0.0, min(1.0, final_score)), 4)


# ---------------------------------------------------------------------------
# Task 3 Grader – Hard (20 patients, dynamic waves, critical events)
# ---------------------------------------------------------------------------

def grade_task3(
    episode_log: List[EpisodeLogEntry],
    expected_wave1_patients: int = 12,
    expected_wave2_patients: int = 8,
    critical_event_step: int = 10,
) -> float:
    """
    Grade a Task 3 (hard) episode.

    Scoring:
    - 50%: Triage accuracy across all patients
    - 15%: Critical patient prioritisation (L1/L2 triaged before L4/L5)
    - 15%: Wave 2 responsiveness (triages after step 10)
    - 10%: Resource efficiency (no violations)
    - 10%: Completion

    Returns: score ∈ [0.0, 1.0]
    """
    if not episode_log:
        return 0.0

    triage_entries = _triage_entries(episode_log)
    n_triaged = len(triage_entries)
    expected_total = expected_wave1_patients + expected_wave2_patients

    if n_triaged == 0:
        return 0.0

    # Accuracy component
    avg_level_acc = sum(e.level_accuracy or 0.0 for e in triage_entries) / n_triaged
    avg_path_acc = sum(e.pathway_accuracy or 0.0 for e in triage_entries) / n_triaged
    accuracy_score = 0.60 * avg_level_acc + 0.40 * avg_path_acc

    # Critical patient prioritisation:
    # Check that high-reward triage entries (≥0.7) appear early in the episode
    high_reward = [e for e in triage_entries if (e.level_accuracy or 0) >= 0.5]
    low_reward = [e for e in triage_entries if (e.level_accuracy or 0) < 0.5]

    if high_reward and low_reward:
        avg_critical_step = sum(e.step for e in high_reward) / len(high_reward)
        avg_noncritical_step = sum(e.step for e in low_reward) / len(low_reward)
        prioritisation_score = 1.0 if avg_critical_step < avg_noncritical_step else 0.5
    elif high_reward:
        prioritisation_score = 1.0
    else:
        prioritisation_score = 0.0

    # Wave 2 responsiveness: triages after critical_event_step
    wave2_entries = [e for e in triage_entries if e.step > critical_event_step]
    if expected_wave2_patients > 0:
        wave2_score = min(1.0, len(wave2_entries) / expected_wave2_patients)
    else:
        wave2_score = 1.0

    # Resource efficiency
    violations = sum(1 for e in triage_entries if e.penalty > 0.05)
    resource_score = max(0.0, 1.0 - violations / max(1, n_triaged))

    # Completion
    completion = min(1.0, n_triaged / expected_total)

    final_score = (
        0.50 * accuracy_score
        + 0.15 * prioritisation_score
        + 0.15 * wave2_score
        + 0.10 * resource_score
        + 0.10 * completion
    )
    return round(max(0.0, min(1.0, final_score)), 4)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def grade_episode(
    task_id: int,
    episode_log: List[EpisodeLogEntry],
) -> float:
    """
    Grade an episode for the given task_id.

    Args:
        task_id: 0=easy, 1=medium, 2=hard
        episode_log: List of EpisodeLogEntry from the completed episode

    Returns:
        score ∈ [0.0, 1.0]
    """
    if task_id == 0:
        return grade_task1(episode_log)
    elif task_id == 1:
        return grade_task2(episode_log)
    elif task_id == 2:
        return grade_task3(episode_log)
    else:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 0, 1, or 2.")
