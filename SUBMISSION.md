# Hackathon Submission Notes (Clinical Triage Agent)

This repository implements an OpenEnv-compatible clinical triage simulation where an agent must triage patients under partial observability and resource constraints.

## What The Project Does

- The environment exposes `reset(task_id, seed)` and `step(action)` via a FastAPI server.
- Observations contain a queue of patients with visible fields (chief complaint, vitals, arrival mode) plus an incremental `revealed_info` dict.
- Agents act using one of: `ASK`, `TRIAGE`, `ESCALATE`, `NO_OP`.
- Rewards are dense and returned on each step; `TRIAGE` is the main scoring action.

Core files:
- `env/triage_env.py`: environment dynamics and task configs.
- `env/models.py`: action/observation schemas.
- `env/reward.py`: reward function (level accuracy, pathway accuracy, speed, resource adherence).
- `env/graders.py`: task graders and episode grading.

## Required: 3 Tasks With Graders

The submission provides 3 tasks, each with an explicit grader:

1. **Easy** (`task_id=0`)
   - Patients: 3
   - Budget: 30
   - Resources: relaxed
   - Grader: `env.graders.grade_easy`

2. **Medium** (`task_id=1`)
   - Patients: 8
   - Budget: 60
   - Resources: tighter; encourages selective `ASK`
   - Grader: `env.graders.grade_medium`

3. **Hard** (`task_id=2`)
   - Patients: 20 (+ dynamic wave mid-episode)
   - Budget: 150
   - Resources: constrained; requires interrupt handling and re-triage to reallocate
   - Grader: `env.graders.grade_hard`

Task configs live in `env/triage_env.py` as `TASK_CONFIGS`. Task metadata also exists in `openenv.yaml` for OpenEnv runners.

## How Grading Works

- **Step-level reward:** for `TRIAGE`, computed in `env/reward.py` via `compute_triage_reward(...)`.
- **Episode score (final_score):** computed in `env/graders.py` via `grade_episode(...)` as the mean completion reward per patient (last `TRIAGE` preferred, otherwise `ESCALATE`).

This produces a single scalar score strictly in `(0, 1)` which is commonly required by validators/judges.

## How To Run (Local)

1. Install deps:
   - `pip install -r requirements.txt`
2. Start the OpenEnv HTTP server:
   - `python -m server.app`
3. Run an evaluation episode:
   - `python inference.py --task_id 2 --env_url http://localhost:7860`
4. Run all tasks (reproducible, no LLM required):
   - `python inference.py --all_tasks --no_llm --seed 42`
