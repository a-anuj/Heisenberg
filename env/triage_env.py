"""
Core TriageEnvironment for the Clinical Triage Agent.

Implements the OpenEnv Environment interface with:
  - reset(task_id, seed)  → TriageObservation
  - step(action)          → (observation, reward, done, info)
  - state property        → EnvironmentState

Supports all 3 task levels (easy/medium/hard) with partial observability,
dynamic patient waves (hard), and deterministic seeded generation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .generators import VALID_QUESTION_KEYS, generate_patients
from .graders import grade_episode, TASK_GRADERS
from .models import (
    ActionType,
    EpisodeLogEntry,
    EnvironmentState,
    Patient,
    PatientSummary,
    Resources,
    TriageAction,
    TriageLevel,
    TriageObservation,
    TriagePathway,
)
from .reward import (
    compute_ask_reward,
    compute_escalate_reward,
    compute_no_op_reward,
    compute_triage_reward,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    0: {  # easy
        "name": "easy",
        "budget": 30,
        "resources": Resources(
            resus_bays_total=3,
            majors_beds_total=10,
            specialists_total=5,
        ),
        "expected_patients": 3,
        "wave_at_step": None,
    },
    1: {  # medium
        "name": "medium",
        "budget": 60,
        "resources": Resources(
            resus_bays_total=2,
            majors_beds_total=6,
            specialists_total=3,
        ),
        "expected_patients": 8,
        "wave_at_step": None,
    },
    2: {  # hard
        "name": "hard",
        "budget": 150,
        "resources": Resources(
            resus_bays_total=3,
            majors_beds_total=10,
            specialists_total=4,
        ),
        "expected_patients": 20,
        "wave_at_step": 10,
    },
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class TriageEnvironment(Environment):
    """
    Emergency Department Triage Environment.

    Simulates real-world ED triage with partial observability,
    resource constraints, and multi-turn decision-making.

    Supported tasks:
        task_id=0 (easy)   – 3 patients, relaxed resources
        task_id=1 (medium) – 8 patients, limited ICU
        task_id=2 (hard)   – 20 patients, dynamic waves, critical event

    Example:
        >>> env = TriageEnvironment()
        >>> obs = env.reset(task_id=0, seed=42)
        >>> action = TriageAction(type="TRIAGE", patient_id="PT-001",
        ...                       level=2, pathway="majors")
        >>> obs, reward, done, info = env.step(action)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initialise the environment (no task loaded yet)."""
        self._state: Optional[EnvironmentState] = None
        self._openenv_state = State(episode_id=str(uuid.uuid4()), step_count=0)
        # Track steps per patient (for speed scoring)
        self._patient_step_start: Dict[str, int] = {}

    # -----------------------------------------------------------------------
    # OpenEnv Interface Methods
    # -----------------------------------------------------------------------

    def reset(  # type: ignore[override]
        self,
        task_id: int = 0,
        seed: Optional[int] = None,
    ) -> TriageObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: 0=easy, 1=medium, 2=hard
            seed: Random seed for reproducibility

        Returns:
            Initial TriageObservation
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Invalid task_id {task_id}. Must be 0, 1, or 2.")

        config = TASK_CONFIGS[task_id]
        patients = generate_patients(task_id=task_id, seed=seed)

        self.task_id = task_id
        self._state = EnvironmentState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            seed=seed,
            patients=patients,
            resources=config["resources"].model_copy(deep=True),
            budget=config["budget"],
        )
        self._openenv_state = State(episode_id=self._state.episode_id, step_count=0)
        self._patient_step_start = {
            p.visible.patient_id: 0 for p in patients
        }

        logger.info(
            "Episode reset: task=%s seed=%s patients=%d budget=%d",
            config["name"],
            seed,
            len(patients),
            config["budget"],
        )

        return self._build_observation(
            last_action_result="Environment reset. Ready to triage.",
            reward=0.01,
        )

    def step(  # type: ignore[override]
        self,
        action: TriageAction | Dict[str, Any] | str,
    ) -> Tuple[TriageObservation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: TriageAction or raw dict conforming to TriageAction schema (or JSON str)

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        # Add robust parsing layer
        error_msg = None
        if isinstance(action, str):
            try:
                import json
                action = json.loads(action)
            except Exception:
                obs = self._build_observation()
                return (obs, 0.01, False, {"error": "invalid_json"})

        if isinstance(action, dict):
            try:
                action = TriageAction(**action)
            except Exception as e:
                obs = self._build_observation()
                return (obs, 0.01, False, {"error": f"invalid_action_schema: {e}"})

        # Wrap step logic in try/except for strict safety check
        try:
            self._state.step_count += 1
            self._openenv_state.step_count = self._state.step_count
            self._state.budget -= 1

            logger.info(
                "step=%d action=%s patient_id=%s budget=%d",
                self._state.step_count,
                action.type,
                action.patient_id,
                self._state.budget,
            )

            # Trigger hard-task wave
            critical_event = None
            wave_step = TASK_CONFIGS[self._state.task_id].get("wave_at_step")
            if (
                wave_step is not None
                and self._state.step_count == wave_step
                and not self._state.wave_triggered
            ):
                self._state.wave_triggered = True
                wave_patients = generate_patients(
                    task_id=self._state.task_id,
                    seed=(self._state.seed or 0) + 999,
                    wave=1,
                )
                for p in wave_patients:
                    self._state.patients.append(p)
                    self._patient_step_start[p.visible.patient_id] = self._state.step_count
                critical_event = (
                    f"MASS CASUALTY EVENT: {len(wave_patients)} new critical patients arrived! "
                    "Re-prioritise immediately."
                )
                self._state.critical_event_triggered = True
                logger.warning("Wave 2 triggered: %d patients added", len(wave_patients))

            # Route to action handler
            reward, result_msg = self._dispatch_action(action)

            # Log entry
            log_entry = self._make_log_entry(action, reward)
            self._state.episode_log.append(log_entry)

            # Check termination
            done = self._check_done()
            self._state.done = done

            obs = self._build_observation(
                last_action_result=result_msg,
                reward=reward,
                critical_event=critical_event,
            )
            obs.done = done
            obs.episode_done = done

            info = {
                "step": self._state.step_count,
                "task_id": self._state.task_id,
                "reward": reward,
                "done": done,
                "budget_remaining": self._state.budget,
                "n_patients_pending": sum(
                    1 for p in self._state.patients
                    if p.visible.triage_status == "pending"
                ),
                "step_score": reward,
                "correct_level": 0,
                "correct_pathway": False,
            }
            
            if action.type == ActionType.TRIAGE and action.patient_id:
                patient = self._find_patient(action.patient_id)
                if patient:
                    info["correct_level"] = patient.ground_truth.level.value
                    try:
                        p_enum = TriagePathway(action.pathway.lower())
                        info["correct_pathway"] = (p_enum == patient.ground_truth.pathway)
                    except Exception:
                        info["correct_pathway"] = False

            if done:
                final_score = self.grade(self._state.task_id, self._state.episode_log)
                info["final_score"] = final_score
                logger.info("Episode done. Final score: %.4f", final_score)

            return obs, reward, done, info

        except Exception as e:
            logger.exception("Error in env step")
            obs = self._build_observation()
            return (obs, 0.01, False, {"error": str(e)})

    def grade(self, task_id: int, log: List[EpisodeLogEntry]) -> float:
        """
        OpenEnv-compatible episode grader.

        Args:
            task_id: The task to grade
            log: List of EpisodeLogEntry objects

        Returns:
            Final episode score in (0, 1)
        """
        return grade_episode(task_id, log)

    @property
    def state(self) -> State:  # type: ignore[override]
        """Return the OpenEnv framework State (episode_id + step_count)."""
        return self._openenv_state

    # -----------------------------------------------------------------------
    # Action Dispatch
    # -----------------------------------------------------------------------

    def _dispatch_action(
        self, action: TriageAction
    ) -> Tuple[float, str]:
        """Route action to specific handler, return (reward, message)."""
        if action.type == ActionType.ASK:
            return self._handle_ask(action)
        elif action.type == ActionType.TRIAGE:
            return self._handle_triage(action)
        elif action.type == ActionType.ESCALATE:
            return self._handle_escalate(action)
        elif action.type == ActionType.NO_OP:
            return self._handle_no_op()
        else:
            return 0.01, f"Unknown action type: {action.type}"

    def _handle_ask(self, action: TriageAction) -> Tuple[float, str]:
        """Handle ASK action: reveal hidden patient information."""
        patient = self._find_patient(action.patient_id)
        if patient is None:
            return 0.01, f"ERROR: Patient {action.patient_id} not found."

        qkey = action.question_key
        if qkey not in VALID_QUESTION_KEYS:
            return 0.01, (
                f"ERROR: Invalid question_key '{qkey}'. "
                f"Valid keys: {sorted(VALID_QUESTION_KEYS)}"
            )

        already_revealed = qkey in patient.visible.revealed_info
        reward = compute_ask_reward(qkey, already_revealed, len(patient.visible.revealed_info))

        if not already_revealed:
            # Reveal the hidden field
            value = self._get_hidden_value(patient, qkey)
            patient.visible.revealed_info[qkey] = value
            msg = f"ASK [{qkey}] for {action.patient_id}: {value}"
        else:
            msg = f"ASK [{qkey}] for {action.patient_id}: already revealed."

        return reward, msg

    def _handle_triage(self, action: TriageAction) -> Tuple[float, str]:
        """Handle TRIAGE action: assign level and pathway to a patient."""
        patient = self._find_patient(action.patient_id)
        if patient is None:
            return 0.01, f"ERROR: Patient {action.patient_id} not found."

        is_retriage = patient.visible.triage_status == "triaged"
        old_pathway = None
        if is_retriage:
            # We allow re-triage for resource management (Task 3 requirement)
            try:
                old_pathway = TriagePathway(patient.triaged_pathway)
            except Exception:
                old_pathway = None

        if action.level is None or action.pathway is None:
            return 0.01, "ERROR: TRIAGE action requires 'level' and 'pathway'."

        resources = self._state.resources
        # 1. If re-triage, release OLD resources first to make room for new assignment
        if is_retriage and old_pathway:
            if old_pathway == TriagePathway.RESUS:
                resources.resus_bays_used = max(0, resources.resus_bays_used - 1)
            elif old_pathway == TriagePathway.MAJORS:
                resources.majors_beds_used = max(0, resources.majors_beds_used - 1)

        # 2. Compute reward with cleaned-up resources
        start_step = self._patient_step_start.get(action.patient_id, 0)
        steps_for_patient = self._state.step_count - start_step

        # Step reward via explicit grader
        grader = TASK_GRADERS[self.task_id]
        reward = grader(
            action,
            patient,
            steps_for_patient,
            resources.resus_available,
            resources.majors_available,
            self._state.escalation_count,
            self.task_id,
        )

        # Update patient state
        try:
            patient.triaged_level = TriageLevel(action.level)
            patient.triaged_pathway = TriagePathway(action.pathway.lower())
        except ValueError:
            pass

        patient.visible.triage_status = "triaged"
        patient.triage_step = self._state.step_count

        # Update resource utilisation
        try:
            pathway = TriagePathway(action.pathway.lower())
            if pathway == TriagePathway.RESUS and resources.resus_available > 0:
                resources.resus_bays_used += 1
            elif pathway in (TriagePathway.MAJORS,) and resources.majors_available > 0:
                resources.majors_beds_used += 1
        except ValueError:
            pass

        msg = (
            f"TRIAGE {action.patient_id}: Level {action.level} → {action.pathway}. "
            f"reward={reward:.3f}"
        )
        return reward, msg

    def _handle_escalate(self, action: TriageAction) -> Tuple[float, str]:
        """Handle ESCALATE action: escalate patient to senior clinician."""
        patient = self._find_patient(action.patient_id)
        if patient is None:
            return 0.01, f"ERROR: Patient {action.patient_id} not found."

        patient.escalation_count += 1
        patient.visible.triage_status = "escalated"
        self._state.escalation_count += 1

        reward = compute_escalate_reward(patient, self._state.escalation_count)
        msg = (
            f"ESCALATE {action.patient_id}: escalated to senior clinician "
            f"(total escalations: {self._state.escalation_count}). "
            f"reward={reward:.3f}"
        )
        return reward, msg

    def _handle_no_op(self) -> Tuple[float, str]:
        """Handle NO_OP: do nothing."""
        reward = compute_no_op_reward()
        return reward, f"NO_OP: step {self._state.step_count}. reward={reward:.3f}"

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _find_patient(self, patient_id: Optional[str]) -> Optional[Patient]:
        """Find a patient by ID in current patient list."""
        if patient_id is None:
            return None
        for p in self._state.patients:
            if p.visible.patient_id == patient_id:
                return p
        return None

    def _get_hidden_value(self, patient: Patient, key: str) -> Any:
        """Retrieve a specific hidden field value."""
        hidden = patient.hidden
        if key == "pain_scale":
            return hidden.pain_scale
        elif key == "duration":
            return f"{hidden.symptom_duration_minutes} minutes"
        elif key == "history":
            return hidden.medical_history or ["None"]
        elif key == "medications":
            return hidden.current_medications or ["None"]
        return None

    def _build_observation(
        self,
        last_action_result: Optional[str] = None,
        reward: float = 0.01,
        critical_event: Optional[str] = None,
    ) -> TriageObservation:
        """Build a TriageObservation from the current state."""
        if self._state is None:
            raise RuntimeError("State not initialised.")

        patient_summaries = [
            PatientSummary(
                patient_id=p.visible.patient_id,
                age=p.visible.age,
                chief_complaint=p.visible.chief_complaint,
                vitals=p.visible.vitals,
                arrival_mode=p.visible.arrival_mode,
                arrival_step=p.visible.arrival_step,
                triage_status=p.visible.triage_status,
                revealed_info=p.visible.revealed_info,
            )
            for p in self._state.patients
        ]

        return TriageObservation(
            patient_queue=patient_summaries,
            resources=self._state.resources,
            step_count=self._state.step_count,
            budget_remaining=self._state.budget,
            last_action_result=last_action_result,
            critical_event=critical_event,
            episode_done=self._state.done,
            done=self._state.done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "task_id": self._state.task_id,
                "task_name": TASK_CONFIGS[self._state.task_id]["name"],
                "n_pending": sum(
                    1 for p in self._state.patients
                    if p.visible.triage_status == "pending"
                ),
            },
        )

    def _check_done(self) -> bool:
        """Check if the episode is complete."""
        if self._state is None:
            return True
        # Done when budget exhausted
        if self._state.budget <= 0:
            return True
        # Done when all patients triaged or escalated
        pending = [
            p for p in self._state.patients
            if p.visible.triage_status == "pending"
        ]
        if not pending:
            # Special case for Task 3: don't end if a future wave is expected
            wave_step = TASK_CONFIGS[self._state.task_id].get("wave_at_step")
            if wave_step and self._state.step_count < wave_step and not self._state.wave_triggered:
                return False
            return True
        return False

    def _make_log_entry(
        self,
        action: TriageAction,
        reward: float,
    ) -> EpisodeLogEntry:
        """Build a log entry for the current step."""
        entry = EpisodeLogEntry(
            step=self._state.step_count,
            action=action.model_dump(exclude_none=True),
            reward=reward,
            patient_id=action.patient_id,
            action_type=action.type.value,
        )

        # Enrich with reward components for graders if TRIAGE
        if action.type == ActionType.TRIAGE and action.patient_id:
            patient = self._find_patient(action.patient_id)
            if patient and action.level and action.pathway:
                from .reward import (
                    compute_level_accuracy,
                    compute_pathway_accuracy,
                    compute_speed_score,
                    compute_resource_adherence,
                )
                start_step = self._patient_step_start.get(action.patient_id, 0)
                steps_used = self._state.step_count - start_step
                entry.level_accuracy = compute_level_accuracy(
                    action.level, patient.ground_truth.level
                )
                entry.pathway_accuracy = compute_pathway_accuracy(
                    action.pathway, patient.ground_truth.pathway
                )
                entry.speed_score = compute_speed_score(steps_used)
                entry.resource_adherence = compute_resource_adherence(
                    action.level,
                    action.pathway,
                    self._state.resources.resus_available,
                    self._state.resources.majors_available,
                )
                # Decode penalty from total reward
                expected_base = (
                    0.40 * entry.level_accuracy
                    + 0.30 * entry.pathway_accuracy
                    + 0.20 * entry.speed_score
                    + 0.10 * entry.resource_adherence
                )
                entry.penalty = max(0.0, expected_base - reward)

        return entry
