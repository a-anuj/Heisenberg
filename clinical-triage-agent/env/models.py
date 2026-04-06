"""
Data models for the Clinical Triage Agent environment.

Defines all Pydantic models for patients, actions, observations,
and environment state used in the emergency department triage simulation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    ASK = "ASK"
    TRIAGE = "TRIAGE"
    ESCALATE = "ESCALATE"
    NO_OP = "NO_OP"


class TriageLevel(int, Enum):
    RESUSCITATION = 1   # Immediate life threat
    EMERGENT = 2        # High risk / severe pain
    URGENT = 3          # Moderate risk / stable
    LESS_URGENT = 4     # Low risk
    NON_URGENT = 5      # Routine


class TriagePathway(str, Enum):
    RESUS = "resus"
    MAJORS = "majors"
    MINORS = "minors"
    FAST_TRACK = "fast_track"
    AMBULATORY = "ambulatory"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Vitals(BaseModel):
    """Observable vital signs for a patient."""

    heart_rate: int = Field(description="Heart rate in bpm", ge=0, le=300)
    spo2: float = Field(description="Oxygen saturation percentage", ge=0.0, le=100.0)
    respiratory_rate: int = Field(description="Respiratory rate breaths/min", ge=0, le=100)
    systolic_bp: Optional[int] = Field(
        default=None, description="Systolic blood pressure mmHg (may be unavailable)", ge=0, le=300
    )
    temperature: Optional[float] = Field(
        default=None, description="Body temperature Celsius (may be unavailable)", ge=30.0, le=43.0
    )


class PatientHidden(BaseModel):
    """Hidden fields revealed only via ASK actions."""

    pain_scale: int = Field(description="Pain scale 0-10", ge=0, le=10)
    symptom_duration_minutes: int = Field(description="Duration of symptoms in minutes", ge=0)
    medical_history: List[str] = Field(default_factory=list, description="Relevant past medical history")
    current_medications: List[str] = Field(default_factory=list, description="Current medications")


class GroundTruth(BaseModel):
    """Ground truth triage decision for scoring (never exposed to agent)."""

    level: TriageLevel
    pathway: TriagePathway
    rationale: str = Field(description="Clinical rationale for triage decision")


class PatientVisible(BaseModel):
    """Visible patient fields presented in observations."""

    patient_id: str
    age: int = Field(ge=0, le=120)
    chief_complaint: str
    vitals: Vitals
    arrival_mode: str = Field(description="walk-in / ambulance / unknown")
    arrival_step: int = Field(default=0, description="Step at which patient arrived")
    triage_status: str = Field(
        default="pending",
        description="pending / triaged / escalated",
    )
    revealed_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information revealed through ASK actions",
    )


class Patient(BaseModel):
    """Full patient model including hidden fields (internal use only)."""

    visible: PatientVisible
    hidden: PatientHidden
    ground_truth: GroundTruth
    escalation_count: int = Field(default=0, description="Number of times escalated")
    triage_step: Optional[int] = Field(default=None, description="Step at which triaged")
    triaged_level: Optional[TriageLevel] = Field(default=None)
    triaged_pathway: Optional[TriagePathway] = Field(default=None)


class Resources(BaseModel):
    """Available and used department resources."""

    resus_bays_total: int = Field(description="Total resuscitation bays", ge=0)
    resus_bays_used: int = Field(default=0, description="Currently occupied resus bays", ge=0)
    majors_beds_total: int = Field(description="Total majors beds", ge=0)
    majors_beds_used: int = Field(default=0, description="Currently occupied majors beds", ge=0)
    specialists_total: int = Field(description="Total available specialists", ge=0)
    specialists_assigned: int = Field(default=0, description="Specialists currently assigned", ge=0)

    @property
    def resus_available(self) -> int:
        return max(0, self.resus_bays_total - self.resus_bays_used)

    @property
    def majors_available(self) -> int:
        return max(0, self.majors_beds_total - self.majors_beds_used)

    @property
    def specialists_available(self) -> int:
        return max(0, self.specialists_total - self.specialists_assigned)


# ---------------------------------------------------------------------------
# Action Models (Agent Input)
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Action the agent can take in one step.

    Only one of the following mutually exclusive action types:
      - ASK: reveal hidden information about a patient
      - TRIAGE: assign a triage level and pathway to a patient
      - ESCALATE: escalate a patient to senior clinician
      - NO_OP: do nothing this step
    """

    type: ActionType = Field(description="Action type: ASK | TRIAGE | ESCALATE | NO_OP")
    patient_id: Optional[str] = Field(
        default=None,
        description="Target patient ID (required for ASK, TRIAGE, ESCALATE)",
    )
    question_key: Optional[str] = Field(
        default=None,
        description="For ASK: which hidden field to reveal (pain_scale | duration | history | medications)",
    )
    level: Optional[int] = Field(
        default=None,
        description="For TRIAGE: triage level 1-5",
        ge=1,
        le=5,
    )
    pathway: Optional[str] = Field(
        default=None,
        description="For TRIAGE: care pathway (resus | majors | minors | fast_track | ambulatory)",
    )


# ---------------------------------------------------------------------------
# Observation Models (Agent Output)
# ---------------------------------------------------------------------------

class PatientSummary(BaseModel):
    """Patient representation in observation (only visible fields)."""

    patient_id: str
    age: int
    chief_complaint: str
    vitals: Vitals
    arrival_mode: str
    arrival_step: int
    triage_status: str
    revealed_info: Dict[str, Any] = Field(default_factory=dict)


class TriageObservation(BaseModel):
    """
    Observation returned to the agent each step.

    Partial observability: hidden patient fields are only accessible
    after explicit ASK actions that reveal them into `revealed_info`.
    """

    patient_queue: List[PatientSummary] = Field(
        description="Current list of patients awaiting triage"
    )
    resources: Resources = Field(description="Current resource utilization")
    step_count: int = Field(description="Current step within episode")
    budget_remaining: int = Field(
        description="Remaining action budget for this episode"
    )
    last_action_result: Optional[str] = Field(
        default=None,
        description="Result/feedback from the previous action",
    )
    critical_event: Optional[str] = Field(
        default=None,
        description="Any critical event triggered this step (e.g., mass casualty)",
    )
    episode_done: bool = Field(default=False, description="Whether episode is finished")

    # Expose base fields needed by openenv framework
    done: bool = Field(default=False, description="Alias for episode_done")
    reward: Optional[float] = Field(default=None, description="Reward from last action")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal Environment State
# ---------------------------------------------------------------------------

class EpisodeLogEntry(BaseModel):
    """Single log entry for episode replay and grading."""

    step: int
    action: Dict[str, Any]
    reward: float
    patient_id: Optional[str]
    action_type: str
    level_accuracy: Optional[float] = None
    pathway_accuracy: Optional[float] = None
    speed_score: Optional[float] = None
    resource_adherence: Optional[float] = None
    penalty: float = 0.0


class EnvironmentState(BaseModel):
    """Full internal state of the environment (not exposed to agent)."""

    episode_id: str
    task_id: int
    seed: Optional[int]
    step_count: int = 0
    patients: List[Patient] = Field(default_factory=list)
    resources: Resources
    budget: int
    done: bool = False
    episode_log: List[EpisodeLogEntry] = Field(default_factory=list)
    escalation_count: int = 0
    wave_triggered: bool = False  # For hard task wave events
    critical_event_triggered: bool = False
