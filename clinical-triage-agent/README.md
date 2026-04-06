---
title: Clinical Triage Agent Environment
emoji: 🏥
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
base_path: /
tags:
  - openenv
  - healthcare
  - triage
  - rl
---

# Clinical Triage Agent

A production-grade OpenEnv-compliant emergency department triage simulation.
This is not a toy — it simulates a real-world ED where an AI agent must triage patients under uncertainty, partial observability, and strict resource constraints.

## API Endpoints (HF Space)

The environment runs on a standalone FastAPI server compatible with the Hugging Face Spaces environment interface.

### `POST /reset`
Initialises the environment.
```json
{
  "task_id": 0,
  "seed": 42
}
```

### `POST /step`
Executes an action.
```json
{
  "action": {
    "type": "TRIAGE",
    "patient_id": "PT-001",
    "level": 2,
    "pathway": "majors"
  }
}
```

Return schema:
```json
{
  "observation": { ... },
  "reward": 0.85,
  "done": false,
  "info": { ... }
}
```

## Supported Tasks

- **Task 0 (easy)**: 3 patients, relaxed resources, simple complaints
- **Task 1 (medium)**: 8 patients, overlapping symptoms, limited ICU capacity, requires `ASK` actions
- **Task 2 (hard)**: 20 patients, dynamic patient waves (mid-episode), critical events, extreme resource pressure

## Observation Space
Partial observability. Hidden fields (pain scale, duration, history, medications) must be explicitly revealed using the `ASK` action.

## Action Space
1. `ASK`: Request hidden information about a patient
2. `TRIAGE`: Assign a clinical level (1-5) and care pathway
3. `ESCALATE`: Escalate complex patients to a senior clinician
4. `NO_OP`: Do nothing

## Reproducibility
All patient generation and validation is 100% deterministic based on `(task_id, seed)`.
