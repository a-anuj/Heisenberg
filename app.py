"""
Standalone FastAPI application for Hugging Face Spaces deployment.

This is the HF Space entrypoint — a self-contained FastAPI app that
exposes the Clinical Triage Agent environment directly without needing
the openenv WebSocket server infrastructure.

Endpoints:
    POST /reset  – Reset the environment for a new episode
    POST /step   – Execute an action and get observation + reward
    GET  /state  – Get current environment state
    GET  /health – Health check
    GET  /schema – Action and observation schemas
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.triage_env import TriageEnvironment, TASK_CONFIGS
from env.graders import grade_episode
from env.models import TriageAction, TriageObservation, Resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinical-triage-agent")

# ---------------------------------------------------------------------------
# Global environment instance (single-session HF Space)
# ---------------------------------------------------------------------------
_env: Optional[TriageEnvironment] = None


def get_env() -> TriageEnvironment:
    """Get the current environment instance, creating if needed."""
    global _env
    if _env is None:
        _env = TriageEnvironment()
    return _env


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 0
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    episode_id: Optional[str]
    step_count: int
    task_id: Optional[int]
    budget_remaining: Optional[int]
    n_patients: Optional[int]
    n_pending: Optional[int]


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Clinical Triage Agent",
    description=(
        "Production-grade OpenEnv-compliant emergency department triage simulation. "
        "An AI agent triages patients under uncertainty and resource constraints."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "env": "clinical-triage-agent"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """
    Reset the environment for a new episode.
    """

    # Handle missing body (validator case)
    if request is None:
        task_id = 0
        seed = None
    else:
        task_id = request.task_id if request.task_id is not None else 0
        seed = request.seed

    if task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id {task_id}. Must be 0, 1, or 2.",
        )

    env = get_env()

    try:
        obs = env.reset(task_id=task_id, seed=seed)
    except Exception as exc:
        logger.exception("Error during reset")
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(
        "Reset: task_id=%d seed=%s patients=%d",
        task_id,
        seed,
        len(obs.patient_queue),
    )

    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {}
    }

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """
    Execute one action step.

    Action types:
    - ASK:     {"type": "ASK", "patient_id": str, "question_key": str}
    - TRIAGE:  {"type": "TRIAGE", "patient_id": str, "level": int (1-5), "pathway": str}
    - ESCALATE:{"type": "ESCALATE", "patient_id": str}
    - NO_OP:   {"type": "NO_OP"}
    """
    env = get_env()
    if env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )

    try:
        action = TriageAction(**request.action)
        obs, reward, done, info = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Error during step")
        raise HTTPException(status_code=500, detail=str(exc))

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    """Get current environment state."""
    env = get_env()
    if env._state is None:
        return StateResponse(
            episode_id=None,
            step_count=0,
            task_id=None,
            budget_remaining=None,
            n_patients=None,
            n_pending=None,
        )

    s = env._state
    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
        task_id=s.task_id,
        budget_remaining=s.budget,
        n_patients=len(s.patients),
        n_pending=sum(
            1 for p in s.patients if p.visible.triage_status == "pending"
        ),
    )


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """Return action and observation JSON schemas."""
    return {
        "action_schema": TriageAction.model_json_schema(),
        "observation_schema": TriageObservation.model_json_schema(),
        "action_types": ["ASK", "TRIAGE", "ESCALATE", "NO_OP"],
        "question_keys": ["pain_scale", "duration", "history", "medications"],
        "triage_levels": {"1": "Resuscitation", "2": "Emergent", "3": "Urgent", "4": "Less Urgent", "5": "Non-Urgent"},
        "pathways": ["resus", "majors", "minors", "fast_track", "ambulatory"],
        "tasks": [
            {"id": 0, "name": "easy", "patients": 3},
            {"id": 1, "name": "medium", "patients": 8},
            {"id": 2, "name": "hard", "patients": 20},
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
