"""
Explicit grader functions for OpenEnv Phase 2 validation.
Each grader wraps the triage reward logic for its respective task.
"""

from .reward import (
    compute_triage_reward, 
    EPSILON, 
    PERFECT_SCORE, 
    ZERO_SCORE
)

def grade_easy(*args, **kwargs):
    """Explicit grader for Easy task."""
    return compute_triage_reward(*args, **kwargs)[0]

def grade_medium(*args, **kwargs):
    """Explicit grader for Medium task."""
    return compute_triage_reward(*args, **kwargs)[0]

def grade_hard(*args, **kwargs):
    """Explicit grader for Hard task."""
    return compute_triage_reward(*args, **kwargs)[0]

# Mapping task_id to explicit graders
TASK_GRADERS = {
    0: grade_easy,
    1: grade_medium,
    2: grade_hard,
}

# TASKS descriptor for OpenEnv validator
TASKS = [
    {"id": "easy", "task_id": 0, "grader": grade_easy},
    {"id": "medium", "task_id": 1, "grader": grade_medium},
    {"id": "hard", "task_id": 2, "grader": grade_hard},
]

def grade_episode(task_id: int, episode_log=None) -> float:
    """
    Episode-level grading for hackathon/OpenEnv runners.

    The environment provides dense step rewards, but many evaluators expect a
    single scalar episode score in (0, 1).

    Strategy:
    - For each patient, use the last completion action reward:
      TRIAGE (preferred) or ESCALATE.
    - Episode score is the mean over patients seen in the log.
    - Missing completions contribute ZERO_SCORE.
    """
    if not episode_log:
        return ZERO_SCORE

    completion_reward_by_patient = {}
    completion_kind_by_patient = {}
    patients_seen = set()

    for entry in episode_log:
        if isinstance(entry, dict):
            action_type = entry.get("action_type")
            patient_id = entry.get("patient_id")
            reward = entry.get("reward")
        else:
            action_type = getattr(entry, "action_type", None)
            patient_id = getattr(entry, "patient_id", None)
            reward = getattr(entry, "reward", None)

        if patient_id:
            patients_seen.add(patient_id)

        if action_type in ("TRIAGE", "ESCALATE") and patient_id and reward is not None:
            prev_kind = completion_kind_by_patient.get(patient_id)
            if prev_kind == "TRIAGE" and action_type == "ESCALATE":
                continue
            completion_kind_by_patient[patient_id] = action_type
            completion_reward_by_patient[patient_id] = float(reward)

    if not patients_seen:
        return ZERO_SCORE

    total = 0.0
    for pid in patients_seen:
        total += float(completion_reward_by_patient.get(pid, ZERO_SCORE))

    score = total / max(1, len(patients_seen))
    score = max(EPSILON, min(PERFECT_SCORE, score))
    return float(score)


def tasks():
    """Task registry for validators that expect a `tasks()` function."""
    return TASKS
