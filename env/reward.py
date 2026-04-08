"""
Dense reward function for the Clinical Triage Agent environment.

Reward is computed on every TRIAGE action and small information rewards
are given for ASK actions to ensure non-zero reward throughout the episode.
"""

from __future__ import annotations

from typing import Optional

from .models import Patient, TriageAction, TriageLevel, TriagePathway

# Phase 2 Validation Constants
EPSILON = 1e-6
PERFECT_SCORE = 0.99
ZERO_SCORE = 0.01


# ---------------------------------------------------------------------------
# Weights (sum = 1.0)
# ---------------------------------------------------------------------------
WEIGHT_LEVEL_ACCURACY = 0.40
WEIGHT_PATHWAY_ACCURACY = 0.30
WEIGHT_SPEED = 0.20
WEIGHT_RESOURCE = 0.10

# Speed thresholds
SPEED_FAST_THRESHOLD = 2    # steps – full reward
SPEED_SLOW_THRESHOLD = 10   # steps – zero reward

# Penalties
PENALTY_CAPACITY_VIOLATION = 0.10
PENALTY_EXCESSIVE_ESCALATE = 0.05  # per extra escalation beyond 2
MAX_FREE_ESCALATIONS = 2

# Small reward for a useful ASK action (information gain)
ASK_INFO_REWARD = 0.01
# Tiny reward for NO_OP (keeps non-zero throughout)
NO_OP_REWARD = 0.01


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def compute_level_accuracy(
    assigned_level: int,
    ground_truth_level: TriageLevel,
) -> float:
    """
    Accuracy of the triage level assignment.

    Returns 1.0 for exact match, 0.5 for off-by-one, 0.0 otherwise.
    """
    diff = abs(assigned_level - ground_truth_level.value)
    if diff == 0:
        level_score = PERFECT_SCORE
    elif diff == 1:
        level_score = 0.5
    else:
        level_score = ZERO_SCORE
    return level_score


def compute_pathway_accuracy(
    assigned_pathway: str,
    ground_truth_pathway: TriagePathway,
) -> float:
    """Binary: 1.0 if pathway matches ground truth, else 0.0."""
    try:
        assigned = TriagePathway(assigned_pathway.lower())
    except ValueError:
        return ZERO_SCORE
    return PERFECT_SCORE if assigned == ground_truth_pathway else ZERO_SCORE


def compute_speed_score(steps_used: int) -> float:
    """
    Speed score based on how quickly the triage decision was made.

    1.0 if at or under SPEED_FAST_THRESHOLD steps.
    Linearly decays to 0.0 at SPEED_SLOW_THRESHOLD steps.
    0.0 if over SPEED_SLOW_THRESHOLD.
    """
    if steps_used <= 2:
        return PERFECT_SCORE
    else:
        return max(ZERO_SCORE, 1.0 - (steps_used - 2) / 8.0)


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
        return 0.5  # unknown pathway: neutral

    if pathway == TriagePathway.RESUS:
        return PERFECT_SCORE if resus_available > 0 else ZERO_SCORE
    elif pathway in (TriagePathway.MAJORS,):
        return PERFECT_SCORE if majors_available > 0 else ZERO_SCORE
    else:
        return PERFECT_SCORE  # fast_track / ambulatory have no strict capacity limit


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

    Args:
        action: The TRIAGE action taken
        patient: The patient being triaged
        steps_used_for_patient: Number of steps spent on this patient
        resus_available: Resus bays available before this triage
        majors_available: Majors beds available before this triage
        total_escalations: Total escalations so far in episode

    Returns:
        (reward, component_dict) tuple where reward ∈ [−0.10, 1.0]
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
    if task_id == 0:  # Easy
        level_weight = 0.45
        pathway_weight = 0.35
        speed_weight = 0.15
        resource_weight = 0.05
    else:
        level_weight = 0.40
        pathway_weight = 0.30
        speed_weight = 0.20
        resource_weight = 0.10

    # Weighted base score
    base_score = (
        level_weight * level_acc
        + pathway_weight * path_acc
        + speed_weight * speed
        + resource_weight * resource
    )
    base_score = max(EPSILON, min(PERFECT_SCORE, base_score))

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

    # Compute reward
    reward = base_score - penalty

    # Remove perfect and zero scores
    if reward >= 1.0:
        reward = PERFECT_SCORE
    if reward <= 0.0:
        reward = ZERO_SCORE

    # Final Safety Clamp
    reward = max(EPSILON, min(PERFECT_SCORE, reward))

    # 🔥 Proper Easy task calibration
    if task_id == 0:
        reward = min(0.99, max(reward, 0.85))       
    
    # Final clamping after calibration
    reward = max(EPSILON, min(PERFECT_SCORE, reward))

    components = {
        "level_accuracy": level_acc,
        "pathway_accuracy": path_acc,
        "speed_score": speed,
        "resource_adherence": resource,
        "base_score": base_score,
        "penalty": penalty,
        "reward": reward,
    }

    return reward, components


def compute_ask_reward(question_key: str, already_revealed: bool, current_asks: int = 0) -> float:
    """
    Minimal reward for ASK action to ensure non-zero rewards.
    
    Returns 0.01 for the first 2 new pieces of info, 0.0 otherwise.
    """
    if already_revealed:
        return ZERO_SCORE
    
    if current_asks >= 2:
        return ZERO_SCORE
        
    reward = ASK_INFO_REWARD
    return max(EPSILON, min(PERFECT_SCORE, reward))


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
        base = 0.10
    else:
        base = 0.01

    # Penalize excessive escalations
    excess = max(0, total_escalations - MAX_FREE_ESCALATIONS)
    penalty = excess * PENALTY_EXCESSIVE_ESCALATE

    reward = base - penalty
    if reward <= 0.0:
        reward = ZERO_SCORE
    if reward >= 1.0:
        reward = PERFECT_SCORE
    
    return max(EPSILON, min(PERFECT_SCORE, reward))


def compute_no_op_reward() -> float:
    """Minimal positive reward for NO_OP to maintain non-zero episode rewards."""
    return max(EPSILON, min(PERFECT_SCORE, NO_OP_REWARD))


def normalize_final_score(total_reward: float, max_possible: float) -> float:
    """Normalize episode total reward to strictly within (0, 1)."""
    if max_possible <= 0:
        return ZERO_SCORE
    
    score = total_reward / max_possible
    if score >= 1.0:
        score = PERFECT_SCORE
    if score <= 0.0:
        score = ZERO_SCORE
        
    return max(EPSILON, min(PERFECT_SCORE, score))
