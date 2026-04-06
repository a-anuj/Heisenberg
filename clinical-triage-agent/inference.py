"""
Inference script for the Clinical Triage Agent environment.

Uses an OpenAI-compatible client to run a full episode with an LLM agent
that receives triage observations and produces structured actions.

Environment variables:
    API_BASE_URL  - Base URL of the OpenAI-compatible API endpoint
    MODEL_NAME    - Name of the model to use
    HF_TOKEN      - Hugging Face token (used as API key)

Log format (exact):
    [START] task=<task> env=clinical-triage-agent model=<model>
    [STEP]  step=1 action=... reward=0.00 done=false error=null
    [END]   success=true steps=10 score=0.85 rewards=[...]

Usage:
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o \
    HF_TOKEN=hf_... \
    python inference.py --task_id 1 --seed 42 --env_url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TASK_NAMES = {0: "easy", 1: "medium", 2: "hard"}


# ---------------------------------------------------------------------------
# Environment HTTP Client
# ---------------------------------------------------------------------------

class EnvHTTPClient:
    """Simple synchronous HTTP client for the triage environment API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task_id: int, seed: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        r = self.client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.client.post(f"{self.base_url}/step", json={"action": action})
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self.client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# LLM Agent Client
# ---------------------------------------------------------------------------

class LLMTriageAgent:
    """OpenAI-compatible LLM agent for triage decisions."""

    SYSTEM_PROMPT = """You are an expert clinical triage agent operating in an emergency department simulation.

Your goal is to correctly triage all patients while maximizing reward and optimizing resources.

---

### ACTION TYPES
- ASK: Reveal hidden info. {"type": "ASK", "patient_id": "PT-001", "question_key": "pain_scale"}
  question_key options: pain_scale, duration, history, medications
- TRIAGE: Assign level+pathway. {"type": "TRIAGE", "patient_id": "PT-001", "level": 2, "pathway": "majors"}
- ESCALATE: Escalate critical patient. {"type": "ESCALATE", "patient_id": "PT-001"}

---

### CORE RULES
- Each patient must be triaged EXACTLY ONCE.
- Never repeat a patient_id for a TRIAGE action.
- Always triage the most critical untriaged patient first.
- Priority: Level 1 (Immediate) > Level 2–3 (Urgent) > Level 4–5 (Non-urgent).

---

### STRATEGY FOR PARTIAL OBSERVABILITY (TASK 2)
- You have a limit of 2 ASK actions per patient.
- If vitals or chief complaint are ambiguous, use ASK for: 1. duration, 2. pain_scale, 3. history.
- After 2 questions, you MUST TRIAGE the patient. Do not over-ask.
- If symptoms are clear (e.g., Cardiac arrest) and vitals are critically abnormal, TRIAGE immediately.

---

### PATHWAY MAPPING (STRICT)
- Level 1 → "resus"
- Level 2 or 3 → "majors"
- Level 4 or 5 → "fast_track"

### RESOURCE-AWARE ROUTING
- If pathway == "resus" AND resus_available == 0, you MUST downgrade to "majors" to avoid capacity violations.
- Always check the 'resources' object in the observation before assigning 'resus'.

---

### OUTPUT FORMAT
Return only a JSON object for ONE action. Respond with valid JSON only."""

    def __init__(self, api_base_url: str, model: str, api_key: str) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(
            base_url=api_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def decide(self, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Send observation to LLM and get an action."""
        obs_text = json.dumps(observation, indent=2)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Step {step}. Current environment state:\n\n{obs_text}\n\n"
                    "Choose ONE action. Respond with valid JSON only."
                ),
            },
        ]

        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 256,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON action from response
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env_url: str,
    task_id: int,
    seed: Optional[int],
    max_steps: int = 50,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Run a full episode and return results.

    Returns dict with: success, steps, score, rewards, errors
    """
    task_name = TASK_NAMES.get(task_id, str(task_id))

    print(
        f"[START] task={task_name} env=clinical-triage-agent model={MODEL_NAME}",
        flush=True,
    )

    env_client = EnvHTTPClient(env_url)
    agent = None
    if use_llm and HF_TOKEN:
        agent = LLMTriageAgent(API_BASE_URL, MODEL_NAME, HF_TOKEN)

    rewards: List[float] = []
    errors: List[Optional[str]] = []
    ask_counts: Dict[str, int] = {}
    escalated_patients: set[str] = set()
    total_steps = 0
    final_score = 0.0
    success = False

    try:
        # Reset
        obs = env_client.reset(task_id=task_id, seed=seed)
        done = obs.get("done", obs.get("episode_done", False))

        step = 0
        while not done and step < max_steps:
            step += 1
            error_msg = None
            action: Dict[str, Any] = {"type": "NO_OP"}

            # Get action from LLM or fallback heuristic
            if agent is not None:
                try:
                    action = agent.decide(obs, step)
                except Exception:
                    action = _fallback_heuristic_action(obs, ask_counts, escalated_patients)
            else:
                action = _fallback_heuristic_action(obs, ask_counts, escalated_patients)

            # Process action for local state tracking
            if action.get("type") == "ASK" and action.get("patient_id"):
                pid = action["patient_id"]
                ask_counts[pid] = ask_counts.get(pid, 0) + 1
            elif action.get("type") == "ESCALATE" and action.get("patient_id"):
                escalated_patients.add(action["patient_id"])

            # Execute step
            try:
                result = env_client.step(action)
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                obs = result.get("observation", {})
                info = result.get("info", {})
                if "error" in info:
                    error_msg = info["error"]
            except Exception as exc:
                error_msg = str(exc)
                reward = 0.0
                done = True
                info = {}

            rewards.append(reward)
            errors.append(error_msg)
            total_steps = step

            error_str = "null" if error_msg is None else f'"{error_msg}"'

            print(
                f"[STEP] step={step} action={json.dumps(action)} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error_str}",
                flush=True,
            )

        # Get final score from info or state
        final_info = env_client.state()
        final_score_raw = info.get("final_score")
        if final_score_raw is not None:
            final_score = float(final_score_raw)
        else:
            # Compute approximate score from rewards
            final_score = sum(rewards) / max(1, len(rewards))

        success = True

    except Exception as exc:
        print(f"[ERROR] Fatal error: {exc}", file=sys.stderr, flush=True)
        success = False
    finally:
        env_client.close()
        if agent:
            agent.close()

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    print(
        f"[END] success={str(success).lower()} steps={total_steps} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {
        "success": success,
        "steps": total_steps,
        "score": final_score,
        "rewards": rewards,
        "errors": errors,
    }


def _fallback_heuristic_action(
    obs: Dict[str, Any],
    ask_counts: Dict[str, int],
    escalated_patients: set[str],
) -> Dict[str, Any]:
    """
    Improved heuristic fallback action for Task 2.
    """
    queue = obs.get("patient_queue", [])
    pending = [p for p in queue if p.get("triage_status") == "pending"]
    resources = obs.get("resources", {})
    resus_avail = resources.get("resus_bays_total", 0) - resources.get("resus_bays_used", 0)

    if not pending:
        return {"type": "NO_OP"}

    # Evaluate all pending patients and pick best to act on
    # In Task 2, we should prioritize critical vitals first
    patient = pending[0]
    pid = patient["patient_id"]
    vitals = patient.get("vitals", {})
    comp = patient.get("chief_complaint", "").lower()
    
    hr = vitals.get("heart_rate", 80)
    spo2 = vitals.get("spo2", 97.0)
    rr = vitals.get("respiratory_rate", 16)
    rev = patient.get("revealed_info", {})
    
    asks = ask_counts.get(pid, 0)

    # 1. Immediate Critical Symptoms -> resus (No ASK needed)
    is_immediate = (spo2 < 85 or hr > 160 or hr < 30 or rr > 40 or "arrest" in comp or "unconscious" in comp)
    
    # 2. Ambiguity Check: If vitals or complaint suggest severity 1-3 but hidden fields unknown
    needs_info = (asks < 2) and (
        "chest pain" in comp or "shortness of breath" in comp or "abdominal" in comp or
        hr > 110 or hr < 50 or spo2 < 93 or rr > 26
    )

    if needs_info and not is_immediate:
        keys = ["duration", "pain_scale", "history"]
        # Find first key not already in rev
        for k in keys:
            if k not in rev:
                return {"type": "ASK", "patient_id": pid, "question_key": k}

    # 3. ESCALATE as last resort if high risk but low info
    if (asks >= 2) and is_immediate and pid not in escalated_patients and len(escalated_patients) < 1:
        return {"type": "ESCALATE", "patient_id": pid}

    # 4. TRIAGE logic
    if is_immediate:
        level, pathway = 1, "resus"
    elif "chest pain" in comp or "shortness of breath" in comp:
        dur = rev.get("duration", "")
        # Chest pain + high pain -> Level 2
        pain = int(rev.get("pain_scale", 5)) if isinstance(rev.get("pain_scale"), (int, float)) else 5
        if pain > 7:
            level, pathway = 2, "majors"
        else:
            level, pathway = 3, "majors"
    elif spo2 < 94 or hr > 110 or rr > 24:
        level, pathway = 3, "majors"
    elif hr > 95 or "fever" in comp or "pain" in comp:
        level, pathway = 4, "fast_track"
    else:
        level, pathway = 5, "fast_track"

    # Capacity Awareness
    if pathway == "resus" and resus_avail <= 0:
        logger_name = "clinical-triage-agent"
        # Since uvicorn log is separate, we just downgrade
        pathway = "majors"

    return {
        "type": "TRIAGE",
        "patient_id": pid,
        "level": level,
        "pathway": pathway,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a clinical triage agent episode"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Task difficulty: 0=easy, 1=medium, 2=hard",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default="http://localhost:7860",
        help="Base URL of the triage environment server",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Use heuristic fallback instead of LLM agent",
    )
    args = parser.parse_args()

    run_episode(
        env_url=args.env_url,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
