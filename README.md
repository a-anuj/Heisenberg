---
title: Clinical Triage Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

# 🏥 Clinical Triage Agent

A production-grade, OpenEnv-compliant emergency department triage simulation environment for training and evaluating AI agents.

## Motivation

Emergency department (ED) triage is a high-stakes, real-world task where clinicians must rapidly assess arriving patients, assign acuity levels, and route them to appropriate care pathways — all under resource constraints and time pressure. Mistakes in triage can lead to delayed treatment for critical patients or wasted resources on non-urgent cases.

This environment models the full complexity of ED triage:
- **Partial observability** — key patient information (pain scale, symptom duration, medical history) is hidden and must be actively requested
- **Resource constraints** — limited resuscitation bays and major beds that must be managed
- **Dynamic events** — mass casualty waves that force re-prioritization mid-episode
- **Multi-patient management** — triaging 3 to 20+ patients per episode

## Action Space

The agent can take one of 4 action types per step:

| Action | Format | Description |
|--------|--------|-------------|
| `ASK` | `{"type": "ASK", "patient_id": "PT-001", "question_key": "pain_scale"}` | Reveal hidden info (max 2 per patient). Keys: `pain_scale`, `duration`, `history`, `medications` |
| `TRIAGE` | `{"type": "TRIAGE", "patient_id": "PT-001", "level": 2, "pathway": "majors"}` | Assign triage level (1–5) and care pathway |
| `ESCALATE` | `{"type": "ESCALATE", "patient_id": "PT-001"}` | Escalate to senior clinician |
| `NO_OP` | `{"type": "NO_OP"}` | Skip this step |

**Triage Levels:** 1=Resuscitation, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent

**Pathways:** `resus`, `majors`, `minors`, `fast_track`, `ambulatory`

## Observation Space

Each step returns a `TriageObservation` containing:

- `patient_queue` — List of patients with visible fields (age, chief complaint, vitals, arrival mode, triage status, revealed info)
- `resources` — Current resource utilization (resus bays, majors beds, specialists)
- `step_count` — Current step in the episode
- `budget_remaining` — Remaining action budget
- `critical_event` — Any mass casualty event triggered this step
- `last_action_result` — Feedback from the previous action

## Tasks

| Task | Difficulty | Patients | Budget | Key Challenge | Expected Score |
|------|-----------|----------|--------|---------------|----------------|
| 0 | **Easy** | 3 | 30 | Basic triage with all info visible | 0.70 – 0.80 |
| 1 | **Medium** | 8 | 60 | Overlapping symptoms, must use ASK strategically, limited ICU | 0.50 – 0.65 |
| 2 | **Hard** | 20 | 150 | Dynamic patient waves, mass casualty events, resource reallocation | 0.20 – 0.40 |

## Reward Design

Dense, per-step rewards with 4 components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Level Accuracy | 35–50% | Exact match = 1.0, off-by-one = 0.25, worse = 0.0 |
| Pathway Accuracy | 25–35% | Correct pathway = 1.0, wrong = 0.0 |
| Speed | 10–25% | Immediate triage = 1.0, linear decay over steps |
| Resource Adherence | 5–15% | Penalizes assigning to full capacity pathways |

**Task difficulty multipliers** scale rewards down for harder tasks (×1.0 easy, ×0.70 medium, ×0.45 hard).

**Episode grading** combines triage accuracy (60%), patient coverage (25%), and budget efficiency (15%).

## Setup & Running

### Prerequisites
- Python 3.10+
- No GPU required (uses remote LLM inference)

### Installation & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python app.py
# Server runs on http://localhost:7860

# In a new terminal, run the agent
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="<your_api_key>"
python inference.py --all_tasks --seed 42
```

### Docker
```bash
docker build -t clinical-triage-agent .
docker run -p 7860:7860 clinical-triage-agent
```

### API Endpoints
- `POST /reset` — Reset environment for new episode
- `POST /step` — Execute one action
- `GET /state` — Get current state
- `GET /health` — Health check
- `GET /schema` — Action/observation schemas
- `GET /docs` — Interactive API docs

## Baseline Scores

Scores from heuristic fallback agent (seed=42):

| Task | Score Range | Notes |
|------|-----------|-------|
| Easy | 0.70 – 0.80 | Straightforward with visible patient info |
| Medium | 0.50 – 0.65 | Requires strategic ASK usage |
| Hard | 0.20 – 0.40 | Mass casualty + resource management challenge |

## Project Structure

```
├── app.py              # FastAPI server (HF Space entrypoint)
├── inference.py        # Baseline inference script with LLM agent
├── openenv.yaml        # OpenEnv spec configuration
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
├── env/
│   ├── models.py       # Typed Pydantic models
│   ├── triage_env.py   # Core environment logic
│   ├── reward.py       # Dense reward function
│   ├── graders.py      # Episode-level graders
│   └── generators.py   # Deterministic patient generators
├── server/
│   └── app.py          # OpenEnv WebSocket server
└── index.html          # Browser UI
```
