---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# 📧 Email Triage Environment

> **An OpenEnv benchmark** — Train and evaluate AI agents on real-world Level-1 customer support email workflows.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-4B8BBE?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Space-FFD21E?style=flat-square)](https://lazycoder01-team-circle.hf.space)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square)](https://docker.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](https://opensource.org/licenses/Apache-2.0)

---

## 📋 Table of Contents

- [Environment Description](#-environment-description)
- [Why This Matters for Agent Research](#-why-this-matters-for-agent-research)
- [Observation Space](#-observation-space)
- [Action Space](#-action-space)
- [Reward Space](#-reward-space)
- [Task Descriptions](#-task-descriptions)
- [API Endpoints](#-api-endpoints)
- [Setup & Usage](#-setup--usage)
- [Baseline Scores](#-baseline-scores)
- [Project Structure](#-project-structure)
- [OpenEnv Validation](#-openenv-validation)

---

## 🌍 Environment Description

The **Email Triage Environment** simulates a genuine **Level-1 Customer Support (CS) pipeline** — one of the most universal operational tasks in every SaaS company.

Every working day, CS agents receive hundreds of support emails and must:

1. **Categorize** each email into the correct support bucket (Billing / Technical / Sales / General)
2. **Prioritize** a mixed inbox by urgency (High / Medium / Low), weighing customer impact, SLA risk, and business value
3. **Draft** a professional, policy-compliant reply to an escalated, angry customer

Getting this right requires **natural language understanding**, **sentiment detection**, **business policy recall**, and **professional writing** — exactly the skills frontier language models claim to have, but rarely are tested on rigorously.

### Why a real-world benchmark?

Most agent benchmarks today test games, puzzles, or narrow coding tasks. Email triage is something **real teams do under real pressure**, with clear correctness criteria and immediate business impact. Training or evaluating an agent here produces a skill that transfers directly to a production deployment.

---

## 🧠 Why This Matters for Agent Research

| Property | Description |
|---|---|
| **Dense reward signal** | Partial-credit graders give agents feedback at every step, not just sparse end-of-episode signals |
| **Natural curriculum** | Three difficulty levels create a natural progression path for curriculum learning |
| **Deterministic graders** | All scoring is reproducible — no LLM-as-judge, no randomness in evaluation |
| **Production realism** | Real-sounding emails, real customer names, real SLA constraints |
| **Multi-skill evaluation** | Classification → ranking → long-form generation tests a wide ability spectrum |

---

## 👁️ Observation Space

Each call to `reset()` or `step()` returns a `TriageObservation` object:

```python
class TriageObservation(BaseModel):
    task_level:   str          # Active task name (e.g. "easy-categorize")
    instructions: str          # Plain-text task instructions for the agent
    emails:       List[Email]  # The inbox emails to act on
    step_number:  int          # Current step index (0 = initial state)
    done:         bool         # True once the episode has concluded
    last_reward:  float | None # Reward from the previous step (None on reset)
```

### `Email` Object

Each email in the `emails` list contains:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique email ID within the episode (e.g. `"e1"`, `"e2"`) |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text |
| `sender` | `str` | Sender display name and address |
| `timestamp` | `str` | ISO-8601 arrival timestamp |
| `thread_id` | `str \| None` | Conversation thread ID (optional) |

---

## 🎮 Action Space

The agent submits a `TriageAction` with exactly **one field populated** per episode, matching the active task:

```python
class TriageAction(BaseModel):
    categories:     dict[str, str] | None  # email_id → "Billing|Technical|Sales|General"
    priorities:     dict[str, str] | None  # email_id → "High|Medium|Low"
    draft_response: str | None             # Full text of the drafted email reply
```

**HTTP format** — agents send actions as JSON inside the standard `Action.metadata` envelope:

```json
{
  "metadata": {
    "categories": { "e1": "Technical" }
  }
}
```

---

## 🏆 Reward Space

Each step returns a `TriageReward` object:

```python
class TriageReward(BaseModel):
    score:     float                    # Normalized score in [0.0, 1.0]
    feedback:  str                      # Human-readable grader explanation
    breakdown: dict[str, float] | None  # Per-criterion scores (hard task only)
```

Rewards are designed to be **partial-credit** and **non-sparse**:

- Easy task: binary (0.0 or 1.0) — clear right/wrong signal
- Medium task: proportional (0.0 – 1.0) + penalty for critical mis-labels — richer signal
- Hard task: five independent weighted criteria — continuous gradient for policy gradient methods

---

## 📋 Task Descriptions

### Task 1 · `easy-categorize` — ⭐ Easy

**Objective:** Classify a **single** incoming customer support email into exactly one of four categories.

| Category | Applies to |
|---|---|
| `Billing` | Payment issues, invoices, subscription charges, refund requests |
| `Technical` | Bugs, errors, login problems, API failures, product not working |
| `Sales` | Upgrade inquiries, demo requests, pricing or feature questions |
| `General` | Feedback, compliments, unclear intent, anything else |

**Scenario:** A user reports a persistent `401 Unauthorized` error after resetting their password twice — a clear technical authentication failure.

**Grader:** Binary
```
score = 1.0  if predicted == "Technical"
score = 0.0  otherwise
```

**Expected frontier model score:** 0.85 – 1.00

---

### Task 2 · `medium-prioritize` — ⭐⭐ Medium

**Objective:** Assign a priority label to each of **5 mixed support emails** representing a realistic morning inbox.

| Priority | Applies to |
|---|---|
| `High` | Production outages, security incidents, billing disputes, severe complaints |
| `Medium` | Sales leads, moderate complaints, time-sensitive but non-critical |
| `Low` | Feature requests, routine admin, general inquiries |

**Scenario:** Five emails spanning the full urgency spectrum — from an enterprise sales inquiry to a production-down emergency affecting 50,000 users, to a routine invoice PDF request.

**Ground truth priorities:**

| Email | Subject | Correct Priority |
|---|---|---|
| `e1` | Enterprise plan inquiry | `Medium` |
| `e2` | URGENT: All APIs returning 500 | `High` 🔴 critical |
| `e3` | Invoice PDF copy request | `Low` |
| `e4` | Feature request: bulk CSV export | `Low` |
| `e5` | Suspected double billing charge | `High` 🔴 critical |

**Grader:** Proportional with penalty
```
base  = correct_count / 5
penalty = 0.10 × (critical emails labeled "Low")   # e2, e5
score = clamp(base − penalty, 0.0, 1.0)
```

**Expected frontier model score:** 0.45 – 0.70

---

### Task 3 · `hard-draft-response` — ⭐⭐⭐ Hard

**Objective:** Write a complete, professional email reply to an **escalated, angry customer** (Marcus Webb) reporting a missing refund after 11 days.

**Customer email summary:** Cancelled subscription on March 8th, promised a $149 refund, 11 days with no update, three unanswered emails, threatening a chargeback and public 1-star reviews.

**Grader:** Rubric-based (5 independent criteria)

| Criterion | Weight | What is checked |
|---|---|---|
| Apology / Empathy | 25% | Contains "sorry", "apologize", "apologise", or "our apologies" |
| Timeline accuracy | 25% | States the "7–10 business days" processing window |
| Policy link | 25% | Includes `https://support.company.com/refunds` |
| Professional tone | 15% | Avoids defensive phrases like "policy says", "not our fault" |
| Personalization | 10% | Addresses the customer by name ("Marcus") |

```
score = Σ weights of passed criteria   ∈ [0.0, 1.0]
```

**Expected frontier model score:** 0.30 – 0.55

---

## 🔌 API Endpoints

Once deployed, the environment exposes a standard OpenEnv HTTP interface at port `7860`:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Health check — returns `{"status": "ok"}` |
| `/reset` | `POST` | Start a new episode; optionally pass `{"task_name": "..."}` |
| `/step` | `POST` | Submit an agent action and receive observation + reward |
| `/state` | `GET` | Get the current episode state without advancing it |
| `/ws` | `WebSocket` | Real-time streaming session interface |

**Example: reset to a specific task**
```bash
curl -X POST https://lazycoder01-team-circle.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "medium-prioritize"}'
```

**Example: submit an action**
```bash
curl -X POST https://lazycoder01-team-circle.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "priorities": {
        "e1": "Medium",
        "e2": "High",
        "e3": "Low",
        "e4": "Low",
        "e5": "High"
      }
    }
  }'
```

---

## 🚀 Setup & Usage

### Prerequisites

- Python ≥ 3.10
- `HF_TOKEN` — your Hugging Face / OpenAI API key
- `API_BASE_URL` — LLM inference endpoint (default: HF Serverless Inference)
- `MODEL_NAME` — model to run (default: `Qwen/Qwen2.5-72B-Instruct`)

### Option 1 · Local Python

```bash
# Clone and install
git clone https://huggingface.co/spaces/lazycoder01/team-circle
cd team-circle
pip install -e .

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — run the baseline inference script
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Option 2 · Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  email-triage-env

# Verify
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
```

### Option 3 · Python Client

```python
import asyncio
from client import EmailTriageClient
from models import TriageAction

async def main():
    async with EmailTriageClient(base_url="https://lazycoder01-team-circle.hf.space") as env:
        # Reset to easy task
        obs = await env.reset_typed(task_name="easy-categorize")
        print(obs.instructions)

        # Submit an action
        result = await env.step_typed(
            TriageAction(categories={"e1": "Technical"})
        )
        print(f"Score: {result.metadata['last_reward']}")

asyncio.run(main())
```

### Option 4 · Live Space (no setup required)

The environment is already running at:

**`https://lazycoder01-team-circle.hf.space`**

---

## 📊 Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Serverless Inference:

| Task | Difficulty | Score | Result |
|---|---|---|---|
| `easy-categorize` | ⭐ Easy | **0.95** | ✅ PASS |
| `medium-prioritize` | ⭐⭐ Medium | **0.60** | ✅ PASS |
| `hard-draft-response` | ⭐⭐⭐ Hard | **0.40** | ✅ PASS |
| **Average** | | **0.65** | ✅ |

**To reproduce:**
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Expected runtime: **< 2 minutes** on 2 vCPU / 8 GB RAM.

---

## 🔍 OpenEnv Validation

```bash
pip install openenv-core
openenv validate
# [OK] email-triage-env: Ready for multi-mode deployment
```

All required fields in `openenv.yaml` are verified:

| Field | Value |
|---|---|
| `spec_version` | `1` |
| `type` | `space` |
| `runtime` | `fastapi` |
| `app` | `server.app:app` |
| `port` | `7860` |

**Pre-submission validator (all 3/3 passed):**
```bash
./validate-submission.sh https://lazycoder01-team-circle.hf.space .
# ✅ HF Space is live and responds to /reset
# ✅ Docker build succeeded
# ✅ openenv validate passed
```

---

## 📁 Project Structure

```
email-triage-env/
├── inference.py             # ← Baseline inference script (run this!)
├── env.py                   # Core environment: EmailTriageEnv + graders
├── models.py                # Pydantic models: Email, TriageObservation, TriageAction, TriageReward
├── client.py                # Typed async HTTP client
├── __init__.py              # Package exports
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package config + dependencies
├── Dockerfile               # Production container
├── README.md                # This file
├── validate-submission.sh   # Pre-submission validator script
├── uv.lock                  # Pinned dependency lockfile
└── server/
    ├── app.py               # FastAPI app via openenv-core create_app()
    └── email_environment.py # OpenEnv Environment base class wrapper
```

---

## 📝 License

Apache 2.0 — see `pyproject.toml`.

---

*Built for the OpenEnv Competition — Meta × Hugging Face, 2025.*