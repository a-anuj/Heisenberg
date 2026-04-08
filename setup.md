# Clinical Triage Agent Setup & Testing Guide

This guide ensures your environment is correctly configured to reproduce the benchmark results for the Clinical Triage Agent across Easy, Medium, and Hard tasks.

## 📋 Prerequisites
- **Python**: 3.10 or higher
- **Hardware**: No special GPU requirements (uses remote LLM inference)

## 🛠️ Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Alternative (if you use `uv`): `uv sync`

## 🚀 Running the Simulation

The simulation requires two components running simultaneously: the **FastAPI Server** (environment) and the **Inference Script** (agent).

### 1. Start the Environment Server

Open a terminal and start the backend:
```bash
python3 -m server.app
```
*Note: This runs on `http://127.0.0.1:7860` by default. Do not close this terminal.*

### 2. Run the Agent (Inference)
Open a **new** terminal and set your environment variables before running the benchmark:

```bash
# Set your Groq or OpenAI compatible API details
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="openai/gpt-oss-120b"
export API_KEY="<your_api_key>"
# Backward-compatible fallback also supported:
# export HF_TOKEN="<your_api_key>"

# Run the benchmark for a specific task
# 0 = Easy, 1 = Medium, 2 = Hard
python3 inference.py --task_id 2
```

Run all tasks (recommended for pre-submission):
```bash
python3 inference.py --all_tasks
```

## 🎯 Benchmark Targets

| Task ID | Difficulty | Target Score | Key Focus |
| :--- | :--- | :--- | :--- |
| **0** | Easy | ≥ 0.90 | Basic triage accuracy |
| **1** | Medium | ≥ 0.70 | Selective `ASK` strategy |
| **2** | Hard | ≥ 0.60 | Arrest interrupts & Resource reallocation |

## 📁 Key Project Files

- **`inference.py`**: The main agent decision logic. contains the heuristic fallback and `processed_patients` logic to prevent loops.
- **`env/reward.py`**: Definiton of the dense reward function.
- **`env/triage_env.py`**: Core environment logic, including critical event wave triggers.
- **`app.py`**: FastAPI wrapper for the environment API.

## 🐛 Troubleshooting
- **Infinite Loops**: Ensure `processed_patients` is correctly initialized in your local `inference.py`.
- **Connection Refused**: Verify the server (`app.py`) is running on port 7860 before starting inference.
- **Low Scores**: Check that `API_KEY` is valid and the model has permission to access the `API_BASE_URL`.
