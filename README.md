---
title: Clinical Triage Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Clinical Triage Agent
A production-grade, OpenEnv-compliant clinical triage simulation environment for AI agents.

This environment simulates an emergency department triage desk. An agent must triage patients under partial information and limited resources, using the standard OpenEnv API: `reset()`, `step()`, and `state()`.

## Environment API

The environment exposes HTTP endpoints (FastAPI):
- `POST /reset` with `{ "task_id": 0|1|2, "seed": int|null }`
- `POST /step` with `{ "action": { ... } }`
- `GET /state`

The canonical OpenEnv app is [`server/app.py`](server/app.py) and is referenced by [`openenv.yaml`](openenv.yaml).

## Action Space

All actions are typed by the `TriageAction` Pydantic model:
- `ASK`: `{"type":"ASK","patient_id":"PT-001","question_key":"pain_scale|duration|history|medications"}`
- `TRIAGE`: `{"type":"TRIAGE","patient_id":"PT-001","level":1-5,"pathway":"resus|majors|minors|fast_track|ambulatory"}`
- `ESCALATE`: `{"type":"ESCALATE","patient_id":"PT-001"}`
- `NO_OP`: `{"type":"NO_OP"}`

## Observation Space

Each step returns a structured `TriageObservation` containing:
- `patient_queue`: list of visible patient summaries (chief complaint, vitals, status, and `revealed_info`)
- `resources`: current resource usage (resus bays, majors beds, specialists)
- `step_count`, `budget_remaining`
- `last_action_result`, optional `critical_event`

Hidden fields (pain scale, duration, history, meds) are only revealed through `ASK` and appear in `revealed_info`.

## Tasks (3) + Graders

Tasks are configured in `env/triage_env.py`:
- Task 0 (`easy`): 3 patients, relaxed resources
- Task 1 (`medium`): 8 patients, constrained resources, selective `ASK` matters
- Task 2 (`hard`): 20 patients plus a mass-casualty wave mid-episode; requires interrupt handling and re-triage for resource reallocation

Grading:
- Step-level rewards are computed in `env/reward.py`.
- Final episode score is computed in `env/graders.py` as the mean per-patient completion reward (TRIAGE preferred, otherwise ESCALATE), clamped strictly to `(0, 1)`.

## Baseline Inference Script

`inference.py` runs an episode against the environment server and emits strict structured logs:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

It supports:
- LLM mode (requires `API_BASE_URL`, `MODEL_NAME`, `API_KEY`; `HF_TOKEN` supported as fallback)
- Reproducible heuristic mode: `--no_llm` (recommended for quick checks)

Baseline (heuristic, `--seed 42`):
- Easy: `0.984`
- Medium: `0.776`
- Hard: `0.767`

## Setup & Testing

For installation, configuration, and benchmark testing instructions:
[**setup.md**](setup.md)
