"""
Dense reward function for the Clinical Triage Agent environment.

Reward is computed on every TRIAGE action and small information rewards
are given for ASK actions to ensure non-zero reward throughout the episode.

Designed to produce realistic score ranges:
  - Easy:   0.7 – 0.8  (competent agent)
  - Medium: 0.5 – 0.65 (competent agent)
  - Hard:   0.2 – 0.4  (competent agent)
"""

from __future__ import annotations

from typing import Optional

from .models import Patient, TriageAction, TriageLevel, TriagePathway

# ---------------------------------------------------------------------------
# Weights (sum = 1.0)
# ---------------------------------------------------------------------------
WEIGHT_LEVEL_ACCURACY = 0.40
WEIGHT_PATHWAY_ACCURACY = 0.30
WEIGHT_SPEED = 0.20
WEIGHT_RESOURCE = 0.10

# Speed thresholds
SPEED_FAST_THRESHOLD = 1    # steps – full reward (immediate triage)
SPEED_SLOW_THRESHOLD = 8    # steps – zero reward

# Penalties
PENALTY_CAPACITY_VIOLATION = 0.15
PENALTY_EXCESSIVE_ESCALATE = 0.08  # per extra escalation beyond 2
MAX_FREE_ESCALATIONS = 2

# Small reward for a useful ASK action (information gain)
ASK_INFO_REWARD = 0.02
# Penalty for redundant ASK (already revealed)
ASK_REDUNDANT_PENALTY = 0.01
# Tiny reward for NO_OP (keeps non-zero)
NO_OP_REWARD = 0.01

# Task difficulty multipliers — scale down step rewards for harder tasks
# so that episode averages naturally land in target ranges
TASK_DIFFICULTY_MULTIPLIER = {
    0: 0.75,   # Easy  — slight reduction for realistic range
    1: 0.55,   # Medium — significant reduction
    2: 0.30,   # Hard  — heavy reduction for genuine challenge
}


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def compute_level_accuracy(
    assigned_level: int,
    ground_truth_level: TriageLevel,
) -> float:
    """
    Accuracy of the triage level assignment.

    Returns 1.0 for exact match, 0.25 for off-by-one, 0.0 otherwise.
    Stricter than clinical tolerance — rewards precise triage.
    """
    diff = abs(assigned_level - ground_truth_level.value)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.25
    elif diff == 2:
        return 0.05
    else:
        return 0.0


def compute_pathway_accuracy(
    assigned_pathway: str,
    ground_truth_pathway: TriagePathway,
) -> float:
    """Binary: 1.0 if pathway matches ground truth, else 0.0."""
    try:
        assigned = TriagePathway(assigned_pathway.lower())
    except ValueError:
        return 0.0
    return 1.0 if assigned == ground_truth_pathway else 0.0


def compute_speed_score(steps_used: int) -> float:
    """
    Speed score based on how quickly the triage decision was made.

    1.0 if at or under SPEED_FAST_THRESHOLD steps.
    Linearly decays to 0.0 at SPEED_SLOW_THRESHOLD steps.
    0.0 if over SPEED_SLOW_THRESHOLD.
    """
    if steps_used <= SPEED_FAST_THRESHOLD:
        return 1.0
    elif steps_used >= SPEED_SLOW_THRESHOLD:
        return 0.0
    else:
        return 1.0 - (steps_used - SPEED_FAST_THRESHOLD) / (SPEED_SLOW_THRESHOLD - SPEED_FAST_THRESHOLD)


def compute_resource_adherence(
    assigned_level: int,
    assigned_pathway: str,
    resus_available: int,
    majors_available: int,
) -> float:
    """
    Score based on whether the resource assignment can actually be fulfilled.

    Returns 1.0 if the resource is available, 0.0 if it would violate capacity.
    """
    try:
        pathway = TriagePathway(assigned_pathway.lower())
    except ValueError:
        return 0.0  # unknown pathway: penalise

    if pathway == TriagePathway.RESUS:
        return 1.0 if resus_available > 0 else 0.0
    elif pathway in (TriagePathway.MAJORS,):
        return 1.0 if majors_available > 0 else 0.0
    else:
        return 1.0  # fast_track / ambulatory have no strict capacity limit


# ---------------------------------------------------------------------------
# Main reward computation
# ---------------------------------------------------------------------------

def compute_triage_reward(
    action: TriageAction,
    patient: Patient,
    steps_used_for_patient: int,
    resus_available: int,
    majors_available: int,
    total_escalations: int,
    task_id: int = 0,
) -> tuple[float, dict]:
    """
    Compute the dense reward for a TRIAGE action.

    Returns:
        (reward, component_dict) tuple where reward ∈ [0.0, 1.0]
    """
    assigned_level = action.level
    assigned_pathway = action.pathway or ""

    gt_level = patient.ground_truth.level
    gt_pathway = patient.ground_truth.pathway

    level_acc = compute_level_accuracy(assigned_level, gt_level)
    path_acc = compute_pathway_accuracy(assigned_pathway, gt_pathway)
    speed = compute_speed_score(steps_used_for_patient)
    resource = compute_resource_adherence(
        assigned_level, assigned_pathway, resus_available, majors_available
    )

    # Task-aware weighting
    if task_id == 0:  # Easy — accuracy matters most
        level_weight = 0.50
        pathway_weight = 0.35
        speed_weight = 0.10
        resource_weight = 0.05
    elif task_id == 1:  # Medium — balanced
        level_weight = 0.40
        pathway_weight = 0.30
        speed_weight = 0.20
        resource_weight = 0.10
    else:  # Hard — speed and resource management critical
        level_weight = 0.35
        pathway_weight = 0.25
        speed_weight = 0.25
        resource_weight = 0.15

    # Weighted base score
    base_score = (
        level_weight * level_acc
        + pathway_weight * path_acc
        + speed_weight * speed
        + resource_weight * resource
    )

    # Penalty: capacity violation
    penalty = 0.0
    try:
        pathway_enum = TriagePathway(assigned_pathway.lower())
    except ValueError:
        pathway_enum = None

    if pathway_enum == TriagePathway.RESUS and resus_available <= 0:
        penalty += PENALTY_CAPACITY_VIOLATION

    if pathway_enum == TriagePathway.MAJORS and majors_available <= 0:
        penalty += PENALTY_CAPACITY_VIOLATION

    # Penalty: excessive escalations (beyond free allowance)
    excess_escalations = max(0, total_escalations - MAX_FREE_ESCALATIONS)
    penalty += excess_escalations * PENALTY_EXCESSIVE_ESCALATE

    # Compute raw reward
    raw_reward = base_score - penalty

    # Apply task difficulty multiplier
    difficulty = TASK_DIFFICULTY_MULTIPLIER.get(task_id, 1.0)
    reward = raw_reward * difficulty

    # Clamp to (0, 1) — validator requires scores in (0.01, 0.99)
    reward = max(0.01, min(0.99, reward))

    components = {
        "level_accuracy": level_acc,
        "pathway_accuracy": path_acc,
        "speed_score": speed,
        "resource_adherence": resource,
        "base_score": base_score,
        "penalty": penalty,
        "difficulty_multiplier": difficulty,
        "reward": reward,
    }

    return reward, components


def compute_ask_reward(question_key: str, already_revealed: bool, current_asks: int = 0) -> float:
    """
    Reward for ASK action.

    Small positive for new info (first 2), penalty for redundant asks.
    """
    if already_revealed:
        return 0.01

    if current_asks >= 2:
        return 0.01  # Over-asking wastes budget

    return ASK_INFO_REWARD


def compute_escalate_reward(
    patient: Patient,
    total_escalations: int,
) -> float:
    """
    Reward for ESCALATE action.

    Escalation is warranted for level-1/2 patients with complex hidden fields.
    Excessive escalation is penalised.
    """
    gt_level = patient.ground_truth.level
    # Escalation is clinically appropriate for level 1-2 patients
    if gt_level in (TriageLevel.RESUSCITATION, TriageLevel.EMERGENT):
        base = 0.08
    else:
        base = 0.01  # Escalating non-critical = minimal reward

    # Penalize excessive escalations
    excess = max(0, total_escalations - MAX_FREE_ESCALATIONS)
    penalty = excess * PENALTY_EXCESSIVE_ESCALATE

    reward = base - penalty
    return max(0.01, min(0.99, reward))


def compute_no_op_reward() -> float:
    """NO_OP gives zero reward — wasting budget should not be rewarded."""
    return max(0.01, min(0.99, NO_OP_REWARD))
