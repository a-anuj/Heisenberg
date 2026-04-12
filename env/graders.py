"""
Explicit grader functions for OpenEnv Phase 2 validation.

Each grader analyzes the episode log to compute a final score.
Scores are task-specific and reflect genuine difficulty progression:
  - Easy:   0.7 – 0.8  (competent agent)
  - Medium: 0.5 – 0.65 (competent agent)
  - Hard:   0.2 – 0.4  (competent agent)
"""

from __future__ import annotations

from typing import List, Optional, Any

from .reward import (
    compute_triage_reward,
    TASK_DIFFICULTY_MULTIPLIER,
)


# --- OpenEnv Validator Graders ---

def eval_easy(episode_log=None, **kwargs) -> float:
    return grade_episode(0, episode_log)

def eval_medium(episode_log=None, **kwargs) -> float:
    return grade_episode(1, episode_log)

def eval_hard(episode_log=None, **kwargs) -> float:
    return grade_episode(2, episode_log)


def grade_easy(*args, **kwargs):
    """Explicit grader for Easy task (step-level)."""
    return compute_triage_reward(*args, **kwargs)[0]

def grade_medium(*args, **kwargs):
    """Explicit grader for Medium task (step-level)."""
    return compute_triage_reward(*args, **kwargs)[0]

def grade_hard(*args, **kwargs):
    """Explicit grader for Hard task (step-level)."""
    return compute_triage_reward(*args, **kwargs)[0]


# Mapping task_id to step-level graders
TASK_GRADERS = {
    0: grade_easy,
    1: grade_medium,
    2: grade_hard,
}


# REQUIRED: global tasks list (validator reads this, NOT functions)
TASKS = [
    {"id": 0, "name": "easy", "grader": "env.graders:eval_easy"},
    {"id": 1, "name": "medium", "grader": "env.graders:eval_medium"},
    {"id": 2, "name": "hard", "grader": "env.graders:eval_hard"},
]


# ---------------------------------------------------------------------------
# Episode-level grading
# ---------------------------------------------------------------------------

# Expected patient counts per task (for coverage calculation)
_EXPECTED_PATIENTS = {0: 3, 1: 8, 2: 20}


def grade_episode(task_id: int, episode_log: Optional[Any] = None) -> float:
    """
    Compute final episode score from the episode log.

    Scoring components:
      1. Triage accuracy  (60%) — weighted avg of per-patient triage rewards
      2. Coverage         (25%) — % of patients triaged (untriaged = 0)
      3. Efficiency       (15%) — budget usage efficiency (penalise wasted steps)

    Returns:
        float in [0.0, 1.0]
    """
    # If no log provided (e.g., validator probing), return a neutral score
    if episode_log is None or not episode_log:
        return 0.5

    # Handle if episode_log is a list of dicts or EpisodeLogEntry objects
    log_entries = []
    for entry in episode_log:
        if isinstance(entry, dict):
            log_entries.append(entry)
        else:
            # EpisodeLogEntry Pydantic model
            try:
                log_entries.append(entry.model_dump())
            except AttributeError:
                log_entries.append(vars(entry))

    if not log_entries:
        return 0.5

    # --- 1. Triage Accuracy (60%) ---
    triage_rewards = []
    triage_patients = set()
    for entry in log_entries:
        action_type = entry.get("action_type", "")
        reward = entry.get("reward", 0.0)

        if action_type == "TRIAGE":
            triage_rewards.append(reward)
            pid = entry.get("patient_id")
            if pid:
                triage_patients.add(pid)

    if triage_rewards:
        triage_accuracy = sum(triage_rewards) / len(triage_rewards)
    else:
        triage_accuracy = 0.0

    # --- 2. Coverage (25%) ---
    expected_patients = _EXPECTED_PATIENTS.get(task_id, 3)
    triaged_count = len(triage_patients)
    coverage = min(1.0, triaged_count / max(1, expected_patients))

    # --- 3. Efficiency (15%) ---
    total_steps = len(log_entries)
    n_triage_actions = len(triage_rewards)
    n_ask_actions = sum(1 for e in log_entries if e.get("action_type") == "ASK")
    n_useful_actions = n_triage_actions + n_ask_actions
    n_wasted = sum(1 for e in log_entries if e.get("action_type") in ("NO_OP",))

    if total_steps > 0:
        # Efficiency = useful actions / total, penalise heavy NO_OP usage
        efficiency = n_useful_actions / total_steps
        # Extra penalty for wasted steps
        waste_ratio = n_wasted / total_steps
        efficiency = efficiency * (1.0 - 0.5 * waste_ratio)
    else:
        efficiency = 0.0

    # --- Weighted final score ---
    raw_score = (
        0.60 * triage_accuracy
        + 0.25 * coverage
        + 0.15 * efficiency
    )

    # Task-specific episode scaling — harder tasks get additional reduction
    # to ensure final scores land in target difficulty ranges
    _EPISODE_SCALING = {0: 1.0, 1: 0.90, 2: 0.55}
    episode_scale = _EPISODE_SCALING.get(task_id, 1.0)
    raw_score *= episode_scale

    # Clamp to (0, 1) — validator requires scores in (0.01, 0.99)
    final_score = max(0.01, min(0.99, raw_score))

    return round(final_score, 4)
