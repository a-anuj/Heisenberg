# Clinical Triage Agent (OpenEnv)

A production-grade **clinical triage simulation environment** built using OpenEnv. This project simulates a high-pressure emergency department where an AI agent must manage patient flow, diagnostic uncertainty, and limited hospital resources under dynamic conditions.

## 📌 Project Overview

The Clinical Triage Agent environment challenges AI agents to perform real-world medical decision-making. The simulation covers:

* **Patient Assessment**: Analyzing vitals and demographic data.
* **Partial Observability**: Deciding when to act versus when to gather more data via `ASK` actions.
* **Triage Logic**: Assigning clinical levels (1-5) and pathways (Resus, Majors).
* **Resource Management**: Prioritizing limited recovery beds and critical care bays.
* **Dynamic Events**: Handling multi-wave patient arrivals and mid-episode emergency interruptions.

The system supports three complexity tiers:
- **Easy**: Direct triage with full visibility.
- **Medium**: Partial information requiring targeted questioning.
- **Hard**: Dynamic arrivals, resource constraints, and mandatory re-triage logic.

## 🚀 Live Demo & API
- **Hugging Face Space**: [a-anuj/clinical-triage-agent](https://huggingface.co/spaces/a-anuj/clinical-triage-agent)
- **Production API**: `https://a-anuj-clinical-triage-agent.hf.space`

## ✨ Key Features

* **OpenEnv-Compliant**: Standardized `reset`, `step`, and `state` interface for seamless agent integration.
* **Hybrid Decision Logic**: Advanced LLM integration with a robust rule-based heuristic fallback.
* **Adaptive Questioning**: Domain-aware `ASK` strategy that minimizes clinical uncertainty while maximizing speed scores.
* **Resource-Aware Routing**: Intelligent monitoring of `resus` and `majors` capacity to prevent bottlenecks.
* **Interrupt Handling**: Immediate prioritization of high-acuity events (e.g., paediatric arrests) within strict temporal windows.
* **Deterministic Evaluation**: Reproducible benchmark results with structured JSON logging (`[START]`, `[STEP]`, `[END]`).
* **Dockerized Deployment**: Fully configured for production deployment on Hugging Face Spaces.

## 📂 Project Structure

* `app.py` – Standalone FastAPI server exposing the environment endpoints.
* `env/` – Core simulation logic (reward functions, clinical models, triage rules).
* `inference.py` – Main agent execution and benchmarking script.
* `openenv.yaml` – Environment and task configuration.
* `requirements.txt` – Project dependencies.
* `Dockerfile` – Production deployment configuration.

## ⚡ API Endpoints

The environment exposes a RESTful API for remote agent interaction:

* `POST /reset`: Initializes a new simulation episode for a specific task and seed.
* `POST /step`: Executes the agent's action (`ASK`, `TRIAGE`, `ESCALATE`) and returns the next observation and reward.
* `GET /state`: Retrieves the current internal state of the emergency department.
* `GET /health`: System status and liveness check.

## 🛠️ How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Environment Server**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```

## 🤖 Run Inference

Set your API credentials and run the benchmarking script against the locally running environment:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="openai/gpt-oss-120b"
export HF_TOKEN="your_key"

# Example: Run Easy Task (Triage Accuracy)
python inference.py --env_url http://localhost:7860 --task_id 0

# Example: Run Medium Task (Partial Information)
python inference.py --env_url http://localhost:7860 --task_id 1

# Example: Run Hard Task against Local Server
python inference.py --env_url http://localhost:7860 --task_id 2

# Example: Run Hard Task against Production HF Space
python inference.py --env_url https://a-anuj-clinical-triage-agent.hf.space --task_id 2
```

## 🚀 Hugging Face Deployment

The environment is ready for one-click deployment to **Hugging Face Spaces** using Docker.
- **Standard Port**: Uses `7860` for seamless integration.
- **Public API**: Exposes endpoints for remote agent evaluation.
- **Persistence**: Optimized for high-concurrency simulation steps.

## 📊 Evaluation

Performance is automatically logged and scored based on four clinical pillars:
1. **Triage Accuracy**: Correctness of Assigned clinical level (1-5).
2. **Pathway Correctness**: Appropriate routing to Resus or Majors.
3. **Efficiency**: Step optimization (speed score).
4. **Resource Adherence**: Maintenance of safe capacity limits.

## 💡 Design Highlights

* **Priority-Based Selection**: Heuristic logic that scans the entire patient queue to identify hidden critical cases.
* **Interrupt Windows**: Specific handling for emergency conditions that arrive mid-wave.
* **Anti-Hacking Rewards**: Calibrated `ASK` rewards to prevent "reward farming" while incentivizing clinical curiosity.
* **Re-Triage Mobility**: The system supports updating triage decisions as patient conditions evolve.

---
This project provides a realistic, technically rigorous framework for evaluating AI agents in complex, high-stakes healthcare environments where decisions must be made under uncertainty and resource constraints.
