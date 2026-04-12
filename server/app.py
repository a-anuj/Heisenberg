"""
FastAPI application for the Clinical Triage Agent Environment.
"""

from openenv.core.env_server.http_server import create_app
from env.models import TriageAction, TriageObservation
from env.triage_env import TriageEnvironment

import uvicorn


app = create_app(
    TriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="clinical-triage-agent",
    max_concurrent_envs=4,
)


def main():
    """Entry point for CLI (required for hackathon runner)"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()