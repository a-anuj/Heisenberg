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
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

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
    """OpenAI Python SDK based LLM agent for triage decisions."""

    SYSTEM_PROMPT = """You are an expert clinical triage agent operating in an emergency department simulation.

Your goal: Correct triage while maximizing efficiency and managing critical resources.

---

### ACTION TYPES
- ASK: Reveal hidden info (max 2 per patient). {"type": "ASK", "patient_id": "PT-001", "question_key": "pain_scale"}
- TRIAGE: Assign level+pathway. Supports Re-triage (updating an existing decision).
- ESCALATE: Escalate critical/ambiguous cases.

---

### TASK 3: DYNAMIC EVENTS & RESOURCE MANAGEMENT
- **Interrupt Handling**: When a new batch of patients arrive (Wave 2), immediately scan for Level 1 cases (e.g., paediatric arrest, unconscious). You MUST triage them to 'resus' within 2 steps of arrival.
- **Resource Reallocation**: If 'resus' bays are full (check 'resources' object) and a new Level 1 patient arrives:
  1. Identify an existing patient in 'resus'.
  2. Re-triage them to 'majors' to free up the bay.
  3. In the next step, assign the new critical patient to 'resus'.
- **Adaptive Questioning**: Choose only ONE most relevant question:
  - Chest pain / SOB → "duration"
  - Injury / Pain → "pain_scale"
  - Unclear symptoms → "history"
- Only ask a second question if the first is highly ambiguous.
- Max 1-2 questions per patient TOTAL. Respond with valid JSON for ONE action only. """

    def __init__(self, api_base_url: str, model: str, api_key: str) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)

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

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()

        # Parse JSON action from response
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())

    def close(self) -> None:
        # OpenAI client has no explicit close requirement.
        return None


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

    model_for_log = MODEL_NAME or "none"
    print(f"[START] task={task_name} env=clinical-triage-agent model={model_for_log}", flush=True)

    env_client = EnvHTTPClient(env_url)
    agent = None
    if use_llm and API_KEY and MODEL_NAME:
        agent = LLMTriageAgent(API_BASE_URL, MODEL_NAME, API_KEY)

    rewards: List[float] = []
    errors: List[Optional[str]] = []
    ask_counts: Dict[str, int] = {}
    processed_patients: set[str] = set()
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
                    action = _fallback_heuristic_action(obs, ask_counts, escalated_patients, processed_patients)
            else:
                action = _fallback_heuristic_action(obs, ask_counts, escalated_patients, processed_patients)

            # Process action for local state tracking
            if action.get("type") == "ASK" and action.get("patient_id"):
                pid = action["patient_id"]
                ask_counts[pid] = ask_counts.get(pid, 0) + 1
            elif action.get("type") == "TRIAGE" and action.get("patient_id"):
                processed_patients.add(action["patient_id"])
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
        f"score={final_score:.2f} rewards=[{rewards_str}]",
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
    processed_patients: set[str] = None
) -> Dict[str, Any]:
    """
    Advanced heuristic action for Task 3: Priority queue + Loop prevention.
    """
    if processed_patients is None:
        processed_patients = set()
        
    queue = obs.get("patient_queue", [])
    # Only act on patients NOT in processed_patients
    available = [p for p in queue if p["patient_id"] not in processed_patients]
    
    resources = obs.get("resources", {})
    resus_avail = resources.get("resus_bays_total", 0) - resources.get("resus_bays_used", 0)

    # 1. SCAN AVAILABLE FOR CRITICAL ARRESTS
    resus_needs = []
    for p in available:
        comp = p.get("chief_complaint", "").lower()
        v = p.get("vitals", {})
        spo2 = v.get("spo2", 97.0)
        hr = v.get("heart_rate", 80)
        
        # Determine if they look like Level 1
        is_l1 = (spo2 < 85 or hr > 160 or hr < 35 or "arrest" in comp or "unconscious" in comp)
        status = p.get("triage_status")
        pathway = p.get("triage_pathway") # This might be in the patient summary if triaged
        
        # If they need resus but aren't there
        if is_l1 and (status != "triaged" or pathway != "resus"):
            resus_needs.append(p)

    # 2. HANDLE CRITICAL INTERRUPT
    if resus_needs:
        p = resus_needs[0]
        pid = p["patient_id"]
        
        if resus_avail > 0:
            return {"type": "TRIAGE", "patient_id": pid, "level": 1, "pathway": "resus"}
        else:
            # RESOURCE REALLOCATION: Move someone out of resus to make room
            # Find someone already in resus who can be downgraded
            in_resus = [pt for pt in queue if pt.get("triage_status") == "triaged" and pt.get("triage_pathway") == "resus"]
            if in_resus:
                # Re-triage them to majors
                return {"type": "TRIAGE", "patient_id": in_resus[0]["patient_id"], "level": 2, "pathway": "majors"}
            else:
                # No one to move? best effort
                return {"type": "TRIAGE", "patient_id": pid, "level": 1, "pathway": "majors"}

    # 3. PENDING PATIENTS SCAN (Priority: 2 > 3 > 4)
    pending = [p for p in queue if p.get("triage_status") == "pending"]
    if not pending:
        return {"type": "NO_OP"}

    # Sort pending by severity (heuristic)
    def _severity_rank(pt):
        v = pt.get("vitals", {})
        s = v.get("spo2", 97.0)
        if s < 91: return 0
        if s < 94: return 1
        return 2
    
    pending.sort(key=_severity_rank)
    patient = pending[0]
    pid = patient["patient_id"]
    rev = patient.get("revealed_info", {})
    comp = patient.get("chief_complaint", "").lower()
    v = patient.get("vitals", {})
    hr = v.get("heart_rate", 80)
    spo2 = v.get("spo2", 97.0)
    asks = ask_counts.get(pid, 0)

    # Decision logic: Adaptive ASK selection
    if asks < 1:
        # Initial selective ASK
        if "chest pain" in comp or "breath" in comp:
            if "duration" not in rev:
                return {"type": "ASK", "patient_id": pid, "question_key": "duration"}
        elif "injury" in comp or "pain" in comp or "fall" in comp:
            if "pain_scale" not in rev:
                return {"type": "ASK", "patient_id": pid, "question_key": "pain_scale"}
        else:
            if "history" not in rev:
                return {"type": "ASK", "patient_id": pid, "question_key": "history"}
    elif asks < 2:
        # Second question ONLY if still critical/ambiguous
        if (spo2 < 94 or "chest" in comp) and "history" not in rev:
            return {"type": "ASK", "patient_id": pid, "question_key": "history"}

    # TRIAGE mapping (after 1-2 ASKs)
    if spo2 < 91 or hr > 125:
        level, pathway = 2, "majors"
    elif spo2 < 95 or hr > 105 or "pain" in comp:
        level, pathway = 3, "majors"
    else:
        level, pathway = 4, "fast_track"

    return {
        "type": "TRIAGE",
        "patient_id": pid,
        "level": level,
        "pathway": pathway,
    }



def grader_fn(result) -> float:
    # handle weird validator inputs
    if not isinstance(result, dict):
        return 0.1

    score = result.get("score", 0.0)

    # enforce strict (0,1)
    if score <= 0.0:
        return 0.1
    if score >= 1.0:
        return 0.9

    return float(score)

# REQUIRED: global tasks list (validator reads this, NOT functions)
TASKS = [
    {"id": "easy", "task_id": 0, "grader": grader_fn},
    {"id": "medium", "task_id": 1, "grader": grader_fn},
    {"id": "hard", "task_id": 2, "grader": grader_fn},
]

def tasks():
    """
    REQUIRED by validator: defines evaluation tasks
    """
    return TASKS



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
        default=None,
        choices=[0, 1, 2],
        help="Task difficulty: 0=easy, 1=medium, 2=hard",
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="Run tasks 0, 1, and 2 sequentially (default when --task_id is omitted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    if args.all_tasks or args.task_id is None:
        for tid in (0, 1, 2):
            run_episode(
                env_url=args.env_url,
                task_id=tid,
                seed=args.seed,
                max_steps=args.max_steps,
                use_llm=not args.no_llm,
            )
        return

    run_episode(
        env_url=args.env_url,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()