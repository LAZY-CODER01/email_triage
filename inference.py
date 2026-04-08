#!/usr/bin/env python3
"""
Email Triage Environment — Baseline Inference Script

Runs a language model against all three tasks of the Email Triage
environment and emits structured stdout logs in the mandatory format:

  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Environment variables (all required to be set before running):
  HF_TOKEN      — Hugging Face / API key used for authentication
  API_BASE_URL  — API endpoint for the LLM (default: HF Serverless Inference)
  MODEL_NAME    — Model identifier to use for inference

Run:
  export HF_TOKEN=hf_xxx
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
  python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import environment (works whether run from project root or server/)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import EmailTriageEnv
from models import TriageAction
from score_utils import MIN_OPEN_SCORE, clamp_open_score

# ---------------------------------------------------------------------------
# Configuration — all read from environment variables
# ---------------------------------------------------------------------------
HF_TOKEN: str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "")  # optional: for from_docker_image()
BENCHMARK: str = "email-triage-env"

# Tasks to evaluate (in order)
ALL_TASKS: List[str] = [
    "easy-categorize",
    "medium-prioritize",
    "hard-draft-response",
]

MAX_STEPS: int = 1        # Single-step episodes
SUCCESS_THRESHOLD: float = 0.5
_SCORE_EPS: float = MIN_OPEN_SCORE

# ---------------------------------------------------------------------------
# Mandatory log helpers — do NOT change format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    reward_safe = clamp_open_score(reward)
    # Collapse whitespace/newlines in action for single-line output
    action_safe = " ".join(action.split())[:200]
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward_safe:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{clamp_open_score(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

_SCHEMA_PROPERTIES = json.dumps(TriageAction.model_json_schema()["properties"], indent=2)

_BASE_PROMPT = """
You MUST respond with a valid JSON object matching these exact properties:
{schema}

IMPORTANT: Provide your keys at the very top level of the JSON object. Do NOT wrap your response inside a "properties" key. 
No explanation outside the JSON — just the JSON.
"""

_EASY_SYSTEM = textwrap.dedent("""
    You are an expert customer support triage agent.

    Your job is to categorize the given support email into EXACTLY ONE category:
      Billing   — payment issues, invoices, subscriptions, charges
      Technical — bugs, errors, login issues, API failures, product not working
      Sales     — upgrade inquiries, demo requests, pricing questions
      General   — anything else (feedback, compliments, unclear intent)
""" + _BASE_PROMPT.format(schema=_SCHEMA_PROPERTIES)).strip()

_MEDIUM_SYSTEM = textwrap.dedent("""
    You are an expert customer support triage agent.

    Your job is to assign a PRIORITY to each of the provided support emails:
      High   — production outages, security issues, billing disputes, severe complaints
      Medium — sales leads, moderate complaints, time-sensitive but not critical
      Low    — feature requests, routine admin, general inquiries
""" + _BASE_PROMPT.format(schema=_SCHEMA_PROPERTIES)).strip()

_HARD_SYSTEM = textwrap.dedent("""
    You are a senior customer support specialist drafting a professional email reply.

    Your reply MUST:
    1. Address the customer by their first name.
    2. Open with a sincere apology for the delay.
    3. Explain that refunds take 7-10 business days to process from the cancellation date.
    4. Include this exact URL: https://support.company.com/refunds
    5. Use a calm, professional, non-defensive tone throughout.
    6. (Optional but rewarded) Offer a goodwill gesture such as a discount or credit.
""" + _BASE_PROMPT.format(schema=_SCHEMA_PROPERTIES)).strip()

SYSTEM_PROMPTS: Dict[str, str] = {
    "easy-categorize": _EASY_SYSTEM,
    "medium-prioritize": _MEDIUM_SYSTEM,
    "hard-draft-response": _HARD_SYSTEM,
}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    system_prompt: str,
    user_content: str,
) -> tuple[str, Optional[str]]:
    """
    Call the model and return (raw_text, error_message_or_None).

    Uses JSON mode for reliable structured output.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,          # deterministic for reproducibility
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "{}").strip()
        return text, None
    except Exception as exc:
        return "{}", str(exc)

def _strict_open_01(score: float) -> float:
    """
    Clamp score to the open interval (0, 1).

    Some validators reject task scores of exactly 0.0 or 1.0. Because the
    submission log prints rewards with 2 decimal places, we keep a 0.01 margin
    so serialized values remain strictly within range too.
    """
    return clamp_open_score(score)


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run one complete episode for the given task.
    Emits [START], [STEP], [END] log lines.
    Returns the episode score in [0.0, 1.0].
    """
    env = EmailTriageEnv(task_name=task_name)
    rewards: List[float] = []
    steps_taken = 0
    score = _SCORE_EPS
    success = False
    error_msg: Optional[str] = None
    did_step_log = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()

        # Build user prompt from observation
        emails_json = json.dumps(
            [e.model_dump() for e in obs.emails], indent=2
        )
        user_content = (
            f"Task instructions:\n{obs.instructions}\n\n"
            f"Emails to process:\n{emails_json}"
        )

        system_prompt = SYSTEM_PROMPTS[task_name]
        raw_text, error_msg = call_model(client, system_prompt, user_content)

        # Parse the model response into a TriageAction
        try:
            action_data = json.loads(raw_text)
            action = TriageAction(**action_data)
        except Exception as parse_err:
            error_msg = f"JSON parse error: {parse_err}"
            action = TriageAction()

        # Step the environment
        _obs, reward_obj, done, _info = await env.step(action)

        reward = _strict_open_01(reward_obj.score)
        rewards.append(reward)
        steps_taken = 1

        log_step(
            step=1,
            action=raw_text,
            reward=reward,
            done=done,
            error=error_msg,
        )
        did_step_log = True

        score = reward  # Single-step episode: score = step reward
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Exception in task {task_name}: {exc}", flush=True)
        score = _SCORE_EPS
        # Ensure evaluators that derive the task score from rewards[] never see
        # an empty list (which is often interpreted as 0.0).
        if not did_step_log:
            log_step(
                step=1,
                action="__error__",
                reward=_SCORE_EPS,
                done=True,
                error=error_msg,
            )
            did_step_log = True
            rewards.append(_SCORE_EPS)
            steps_taken = 1

    finally:
        await env.close()
        log_end(
            success=success,
            steps=steps_taken,
            rewards=rewards,
        )

    return _strict_open_01(score)


# ---------------------------------------------------------------------------
# Main: iterate over all tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    if not HF_TOKEN:
        print(
            "[ERROR] No API key found. Set HF_TOKEN or OPENAI_API_KEY.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_scores: Dict[str, float] = {}
    for task_name in ALL_TASKS:
        score = await run_task(client, task_name)
        all_scores[task_name] = score
        print()  # Blank line between tasks for readability

    # Summary table
    print("=" * 50, flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 50, flush=True)
    for task, sc in all_scores.items():
        status = "PASS" if sc >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {task:<30} {sc:.3f}  [{status}]", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  {'AVERAGE':<30} {avg:.3f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
