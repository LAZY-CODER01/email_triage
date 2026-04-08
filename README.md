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

> **OpenEnv** — A real-world environment for training and evaluating AI agents on customer support email workflows.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HF-Space-yellow)](https://huggingface.co)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

---

## 🌍 Environment Description

The **Email Triage Environment** models a genuine Level-1 customer support (CS) pipeline. Every working day, CS agents receive hundreds of support emails and must:

1. **Categorize** each email (Billing / Technical / Sales / General)
2. **Prioritize** a mixed inbox by urgency (High / Medium / Low)
3. **Draft** a polished, policy-compliant response to an unhappy customer

This is a universal real-world task found in every SaaS company. Getting it right requires natural language understanding, sentiment detection, policy recall, and professional writing — making it an excellent benchmark for language model agents.

**Why this matters for RL/agent research:**
- Provides dense (not just sparse) reward signals at each episode step
- Three clearly differentiated difficulty levels create a natural curriculum
- Graders are deterministic and reproducible for fair comparison
- Directly applicable to building production AI customer support agents

---

## 🎮 Action & Observation Spaces

### Observation (`TriageObservation`)

```python
class TriageObservation(BaseModel):
    task_level:   str          # which task is active
    instructions: str          # plain-text instructions for the agent
    emails:       List[Email]  # inbox emails to act on
    step_number:  int          # current step index (0 = initial)
    done:         bool         # True once episode has concluded
    last_reward:  float | None # reward from the previous step
```

Each `Email` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique ID within the episode (e.g. `"e1"`) |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text |
| `sender` | `str` | Sender name / address |
| `timestamp` | `str` | ISO-8601 arrival time |
| `thread_id` | `str?` | Conversation thread ID (optional) |

### Action (`TriageAction`)

Exactly **one** field should be populated per episode, matching the active task:

```python
class TriageAction(BaseModel):
    categories:     dict[str, str] | None  # email_id → "Billing|Technical|Sales|General"
    priorities:     dict[str, str] | None  # email_id → "High|Medium|Low"
    draft_response: str | None             # full text of drafted reply
```

### Reward (`TriageReward`)

```python
class TriageReward(BaseModel):
    score:     float                  # in [0.0, 1.0]
    feedback:  str                    # human-readable explanation
    breakdown: dict[str, float] | None  # per-criterion scores (hard task)
```

---

## 📋 Task Descriptions

### Task 1: `easy-categorize` ⭐ Easy

**Objective:** Classify a single customer email into one of four support buckets.

| Category | Description |
|----------|-------------|
| `Billing` | Payment issues, invoices, subscription charges, refunds |
| `Technical` | Bugs, errors, login problems, API failures |
| `Sales` | Upgrade inquiries, demo requests, pricing questions |
| `General` | Feedback, compliments, unclear intent |

**Grader:** Binary — `1.0` for correct category, `0.0` otherwise.  
**Expected agent score:** 0.85–1.0 (clear-cut case)

---

### Task 2: `medium-prioritize` ⭐⭐ Medium

**Objective:** Assign priority labels to a batch of **5 mixed support emails**.

| Priority | Applies to |
|----------|-----------|
| `High` | Production outages, security issues, billing disputes |
| `Medium` | Sales leads, moderate complaints, time-sensitive but not critical |
| `Low` | Feature requests, routine admin, general inquiries |

**Grader:** Proportional — `correct_count / 5`. Penalty of `-0.10` per critical email
(`e2`: production down, `e5`: billing dispute) that is incorrectly labeled `Low`.  
**Expected agent score:** 0.45–0.70 (requires nuanced priority judgment)

---

### Task 3: `hard-draft-response` ⭐⭐⭐ Hard

**Objective:** Draft a complete, professional email response to an angry customer reporting a missing refund (Marcus Webb, cancelled 11 days ago).

**Grader (5 independent criteria):**

| Criterion | Weight | What's checked |
|-----------|--------|----------------|
| Apology / Empathy | 0.25 | Contains "sorry", "apologize", or equivalent |
| Timeline accuracy | 0.25 | States "7-10 business days" processing time |
| Policy link | 0.25 | Includes `https://support.company.com/refunds` |
| Professional tone | 0.15 | No defensive phrases like "policy says", "not our fault" |
| Personalization | 0.10 | Addresses the customer as "Marcus" |

**Expected agent score:** 0.30–0.55 (requires policy recall + tone control)

---

## 🔌 API Endpoints

Once deployed, the environment exposes a standard OpenEnv HTTP interface:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Health check |
| `/reset` | `POST` | Start a new episode |
| `/step` | `POST` | Submit an action |
| `/state` | `GET` | Get current state |
| `/ws` | `WebSocket` | Real-time session interface |
| `/web` | `GET` | Interactive web UI (set `ENABLE_WEB_INTERFACE=true`) |

**Example: Reset for medium-prioritize task**
```bash
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "medium-prioritize"}'
```

---

## 🚀 Setup & Usage

### Option 1: Python (local)

```bash
# Install
pip install -e .

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Option 2: Docker

```bash
# Build
docker build -t email-triage-env .

# Run
docker run -p 7860:7860 email-triage-env

# Test the health endpoint
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{}'
```

### Option 3: Python client

```python
import asyncio
from client import EmailTriageClient
from models import TriageAction

async def main():
    async with EmailTriageClient(base_url="https://your-space.hf.space") as env:
        obs = await env.reset_typed(task_name="easy-categorize")
        result = await env.step_typed(
            TriageAction(categories={"e1": "Technical"})
        )
        print(result.metadata)

asyncio.run(main())
```

---

## 📊 Baseline Scores

Results using `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Serverless Inference:

| Task | Score | Status |
|------|-------|--------|
| `easy-categorize` | ~0.95 | ✅ PASS |
| `medium-prioritize` | ~0.60 | ✅ PASS |
| `hard-draft-response` | ~0.40 | ✅ PASS |
| **Average** | **~0.65** | ✅ |

To reproduce:
```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

---

## 🔍 OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

All required fields in `openenv.yaml` are present:
- `spec_version: 1`
- `type: space`
- `runtime: fastapi`
- `app: server.app:app`
- `port: 7860`

---

## 📁 Project Structure

```
email-triage-env/
├── __init__.py              # Package exports
├── models.py                # Pydantic: Email, TriageObservation, TriageAction, TriageReward
├── env.py                   # Core logic: EmailTriageEnv, dataset, graders
├── client.py                # EnvClient subclass for typed client access
├── inference.py             # Baseline multi-task inference script
├── openenv.yaml             # OpenEnv manifest (spec_version, type, runtime, ...)
├── pyproject.toml           # Dependencies and package configuration
├── Dockerfile               # Production container definition
├── README.md                # This file
└── server/
    ├── app.py               # FastAPI app (create_app from openenv-core)
    └── email_environment.py # Environment(openenv_base) server-side implementation
```

---

## 📝 License

Apache 2.0 — see `pyproject.toml`.