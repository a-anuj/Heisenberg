"""
FastAPI application for the Clinical Triage Agent Environment.

Exposes the TriageEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv EnvClient.

Endpoints:
    POST /reset  - Reset the environment
    POST /step   - Execute an action
    GET  /state  - Get current environment state
    GET  /schema - Get action/observation schemas
    WS   /ws     - WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core[core]"
    ) from e

try:
    from ..models import TriageAction, TriageObservation
    from ..env.triage_env import TriageEnvironment
except (ModuleNotFoundError, ImportError):
    from models import TriageAction, TriageObservation
    from env.triage_env import TriageEnvironment


app = create_app(
    TriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="clinical-triage-agent",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
