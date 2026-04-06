"""
TriageEnv client for the Clinical Triage Agent environment.

Maintains a persistent WebSocket connection to the environment server,
enabling efficient multi-step triage interactions.
"""

from __future__ import annotations

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TriageAction, TriageObservation, PatientSummary, Vitals, Resources


class TriageEnv(EnvClient[TriageAction, TriageObservation, State]):
    """
    Client for the Clinical Triage Agent Environment.

    Connects via WebSocket to the environment server for low-latency
    multi-step triage episode execution.

    Example:
        >>> with TriageEnv(base_url="http://localhost:7860") as env:
        ...     result = env.reset()
        ...     action = TriageAction(
        ...         type="TRIAGE",
        ...         patient_id=result.observation.patient_queue[0].patient_id,
        ...         level=2,
        ...         pathway="majors"
        ...     )
        ...     result = env.step(action)
        ...     print(result.reward)

    Example with Docker:
        >>> env = TriageEnv.from_docker_image("clinical-triage-agent:latest")
        >>> try:
        ...     result = env.reset()
        ...     result = env.step(TriageAction(type="NO_OP"))
        ... finally:
        ...     env.close()
    """

    def _step_payload(self, action: TriageAction) -> Dict:
        """Convert TriageAction to JSON payload for WebSocket step message."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        """Parse server response into StepResult[TriageObservation]."""
        obs_data = payload.get("observation", {})
        observation = self._parse_observation(obs_data, payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_observation(
        self,
        obs_data: Dict,
        payload: Optional[Dict] = None,
    ) -> TriageObservation:
        """Parse raw observation dict into TriageObservation."""
        payload = payload or {}

        # Parse patient queue
        raw_queue = obs_data.get("patient_queue", [])
        patient_summaries = []
        for p in raw_queue:
            vitals_data = p.get("vitals", {})
            patient_summaries.append(
                PatientSummary(
                    patient_id=p.get("patient_id", ""),
                    age=p.get("age", 0),
                    chief_complaint=p.get("chief_complaint", ""),
                    vitals=Vitals(
                        heart_rate=vitals_data.get("heart_rate", 0),
                        spo2=vitals_data.get("spo2", 0.0),
                        respiratory_rate=vitals_data.get("respiratory_rate", 0),
                        systolic_bp=vitals_data.get("systolic_bp"),
                        temperature=vitals_data.get("temperature"),
                    ),
                    arrival_mode=p.get("arrival_mode", "unknown"),
                    arrival_step=p.get("arrival_step", 0),
                    triage_status=p.get("triage_status", "pending"),
                    revealed_info=p.get("revealed_info", {}),
                )
            )

        # Parse resources
        raw_resources = obs_data.get("resources", {})
        resources = Resources(
            resus_bays_total=raw_resources.get("resus_bays_total", 0),
            resus_bays_used=raw_resources.get("resus_bays_used", 0),
            majors_beds_total=raw_resources.get("majors_beds_total", 0),
            majors_beds_used=raw_resources.get("majors_beds_used", 0),
            specialists_total=raw_resources.get("specialists_total", 0),
            specialists_assigned=raw_resources.get("specialists_assigned", 0),
        )

        return TriageObservation(
            patient_queue=patient_summaries,
            resources=resources,
            step_count=obs_data.get("step_count", 0),
            budget_remaining=obs_data.get("budget_remaining", 0),
            last_action_result=obs_data.get("last_action_result"),
            critical_event=obs_data.get("critical_event"),
            episode_done=payload.get("done", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
