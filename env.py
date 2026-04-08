"""
Email Triage Environment — Core Logic

Implements the OpenEnv step()/reset()/state() contract as a pure Python
class (used both directly and wrapped by the FastAPI server).

Three tasks, each with deterministic graders:
  1. easy-categorize   — classify one email into Billing/Technical/Sales/General
  2. medium-prioritize — rank 5 mixed emails by urgency (High/Medium/Low)
  3. hard-draft-response — write a policy-compliant response to an angry customer
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from models import Email, TriageAction, TriageObservation, TriageReward

# ---------------------------------------------------------------------------
# Canonical email datasets (one per task)
# ---------------------------------------------------------------------------

_EASY_EMAILS = [
    Email(
        id="e1",
        subject="Cannot login to my account",
        body=(
            "Hi, I keep getting a '401 Unauthorized' error whenever I try to "
            "access my dashboard. I've reset my password twice but the problem "
            "persists. My account email is john.doe@example.com. Please help!"
        ),
        sender="John Doe <john.doe@example.com>",
        timestamp="2024-03-15T09:12:00Z",
    )
]

_EASY_SOLUTION: Dict[str, str] = {"e1": "Technical"}

# ---------------------------------------------------------------------------

_MEDIUM_EMAILS = [
    Email(
        id="e1",
        subject="Interested in Enterprise Plan",
        body=(
            "Hello, we are a team of 200 engineers considering upgrading to your "
            "Enterprise tier next quarter. Could you send me a pricing deck and "
            "schedule a demo call? — Sarah, Head of Engineering at Acme Corp."
        ),
        sender="Sarah Lin <sarah.lin@acme.com>",
        timestamp="2024-03-15T08:00:00Z",
    ),
    Email(
        id="e2",
        subject="URGENT: All APIs returning 500 — Production is down!",
        body=(
            "Our entire production environment is down. Every API endpoint is "
            "returning HTTP 500. We have 50,000 active users affected RIGHT NOW. "
            "I need someone on the phone immediately. This started 12 minutes ago. "
            "SLA breach imminent."
        ),
        sender="ops-alerts@bigclient.io",
        timestamp="2024-03-15T09:47:00Z",
    ),
    Email(
        id="e3",
        subject="Invoice #INV-4421 — Can I get a PDF copy?",
        body=(
            "Hi, could you resend invoice number INV-4421 as a PDF? I need it "
            "for our accounting department's records. Thanks!"
        ),
        sender="Bob Accounting <bob@smallbiz.net>",
        timestamp="2024-03-15T09:55:00Z",
    ),
    Email(
        id="e4",
        subject="Feature request: bulk CSV export",
        body=(
            "We'd love a bulk CSV export feature on the analytics dashboard. "
            "Manually downloading 30-day reports page-by-page is a pain. Is "
            "this on your roadmap? Not urgent, just a suggestion."
        ),
        sender="Alice Dev <alice@startup.io>",
        timestamp="2024-03-15T10:02:00Z",
    ),
    Email(
        id="e5",
        subject="Suspected billing error — charged twice this month",
        body=(
            "I think I was double-charged this month. My credit card statement "
            "shows two charges of $299 on March 1st and March 3rd. Please "
            "refund the duplicate charge ASAP. This is really frustrating."
        ),
        sender="Chris Payne <chris.payne@personalmail.com>",
        timestamp="2024-03-15T10:15:00Z",
    ),
]

_MEDIUM_SOLUTION: Dict[str, str] = {
    "e1": "Medium",   # Sales lead — important but not on fire
    "e2": "High",     # Production outage — critical
    "e3": "Low",      # Routine billing admin
    "e4": "Low",      # Feature request — no urgency
    "e5": "High",     # Billing dispute with angry tone — needs fast response
}

# Which emails must NOT be labeled Low (penalized if they are)
_MEDIUM_MUST_NOT_BE_LOW = {"e2", "e5"}

# ---------------------------------------------------------------------------

_HARD_EMAILS = [
    Email(
        id="e1",
        subject="WHERE IS MY REFUND — I'm done with your company",
        body=(
            "Hi,\n\n"
            "My name is Marcus Webb. I cancelled my subscription on March 8th "
            "and I was promised a full refund of $149. It has now been 11 days "
            "and I see NOTHING in my account. I've emailed three times with no "
            "response. This is absolutely unacceptable. I'm going to dispute "
            "the charge with my bank and leave a 1-star review everywhere "
            "unless this is resolved TODAY.\n\n"
            "— Marcus Webb"
        ),
        sender="Marcus Webb <marcus.webb@gmail.com>",
        timestamp="2024-03-19T14:33:00Z",
    )
]

_HARD_INSTRUCTIONS = (
    "Draft a professional, empathetic response to Marcus. Your reply MUST:\n"
    "1. Address Marcus by name and sincerely apologise for the delay.\n"
    "2. Explain that refunds take 7–10 business days to process.\n"
    "3. Include the support policy link: https://support.company.com/refunds\n"
    "4. Remain professional and avoid defensive language.\n"
    "5. (Bonus) Offer a goodwill gesture (e.g. discount, account credit).\n"
)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict] = {
    "easy-categorize": {
        "emails": _EASY_EMAILS,
        "instructions": (
            "Categorize the email below into exactly one of these categories:\n"
            "  Billing | Technical | Sales | General\n\n"
            "Return a JSON object with key 'categories' mapping each email ID "
            "to its category string."
        ),
        "solution": _EASY_SOLUTION,
    },
    "medium-prioritize": {
        "emails": _MEDIUM_EMAILS,
        "instructions": (
            "Assign a priority label to each of the 5 emails below:\n"
            "  High | Medium | Low\n\n"
            "- High:   production issues, security, billing disputes, severe complaints\n"
            "- Medium: sales leads, moderate complaints, time-sensitive but not critical\n"
            "- Low:    routine admin, feature requests, general inquiries\n\n"
            "Return a JSON object with key 'priorities' mapping each email ID "
            "to its priority string."
        ),
        "solution": _MEDIUM_SOLUTION,
    },
    "hard-draft-response": {
        "emails": _HARD_EMAILS,
        "instructions": _HARD_INSTRUCTIONS,
        "solution": {},  # Graded via heuristics
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """
    Clamp a raw score to the open interval (0.0, 1.0) as required by the
    OpenEnv specification.  Scores of exactly 0.0 or 1.0 are rejected by
    the validator; we map them to 0.01 / 0.99 respectively.
    """
    return max(0.01, min(0.99, round(score, 4)))


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _grade_easy(action: TriageAction, solution: Dict[str, str]) -> TriageReward:
    """Binary grader: correct category ≈ 0.99, wrong ≈ 0.01."""
    if not action.categories:
        return TriageReward(
            score=_clamp(0.0),
            feedback="No 'categories' provided. Expected a mapping of email ID → category.",
        )
    predicted = action.categories.get("e1", "").strip().title()
    expected = solution["e1"]
    if predicted == expected:
        return TriageReward(
            score=_clamp(1.0),
            feedback=f"Correct! '{predicted}' is the right category for this email.",
        )
    valid = {"Billing", "Technical", "Sales", "General"}
    if predicted not in valid:
        return TriageReward(
            score=_clamp(0.0),
            feedback=(
                f"Invalid category '{predicted}'. "
                f"Must be one of: {', '.join(sorted(valid))}."
            ),
        )
    return TriageReward(
        score=_clamp(0.0),
        feedback=(
            f"Incorrect. You predicted '{predicted}' but the answer is '{expected}'. "
            "The email describes a login/authentication error — that's Technical support."
        ),
    )


def _grade_medium(action: TriageAction, solution: Dict[str, str]) -> TriageReward:
    """
    Proportional grader: one point per correctly labelled email.
    Penalty of 0.1 per 'must-not-be-Low' email labelled Low.
    Final score clamped to (0.01, 0.99) per OpenEnv spec.
    """
    if not action.priorities:
        return TriageReward(
            score=_clamp(0.0),
            feedback="No 'priorities' provided. Expected a mapping of email ID → priority.",
        )
    total = len(solution)
    correct = 0
    penalty = 0.0
    breakdown: Dict[str, float] = {}

    for eid, expected in solution.items():
        predicted = action.priorities.get(eid, "").strip().title()
        if predicted == expected:
            correct += 1
            breakdown[eid] = 1.0
        else:
            breakdown[eid] = 0.0
            if eid in _MEDIUM_MUST_NOT_BE_LOW and predicted == "Low":
                penalty += 0.1

    base = correct / total
    score = _clamp(max(0.0, min(1.0, base - penalty)))

    details = ", ".join(
        f"{eid}={'✓' if v == 1.0 else '✗'}" for eid, v in sorted(breakdown.items())
    )
    feedback = (
        f"Correctly prioritized {correct}/{total} emails ({details}). "
        f"Base score: {base:.2f}"
    )
    if penalty > 0:
        feedback += f", penalty: -{penalty:.2f} (critical email mis-labeled Low)"

    return TriageReward(score=score, feedback=feedback, breakdown=breakdown)


def _grade_hard(action: TriageAction) -> TriageReward:
    """
    Rubric-based grader with 5 independent sub-criteria.
    Each criterion contributes a fixed weight to the total score.
    """
    response = (action.draft_response or "").strip()
    rl = response.lower()

    breakdown: Dict[str, float] = {}

    # 1. Apology / Empathy (0.25)
    apology_keywords = ["sorry", "apologize", "apologise", "sincerely apologize", "our apologies"]
    breakdown["apology"] = 0.25 if any(kw in rl for kw in apology_keywords) else 0.0

    # 2. Timeline accuracy (0.25) — 7-10 business days
    timeline_keywords = ["7-10 business", "7 to 10 business", "seven to ten business",
                         "7–10 business", "7 - 10 business"]
    breakdown["timeline"] = 0.25 if any(kw in rl for kw in timeline_keywords) else 0.0

    # 3. Policy link (0.25)
    breakdown["policy_link"] = (
        0.25 if "support.company.com/refunds" in response else 0.0
    )

    # 4. Professional tone — penalise defensive / aggressive language (0.15)
    defensive_phrases = ["not our fault", "policy says", "you should have", "rules state",
                         "you need to understand", "actually,", "to be honest,"]
    is_defensive = any(p in rl for p in defensive_phrases)
    breakdown["professional_tone"] = 0.0 if is_defensive else 0.15

    # 5. Personalisation — uses customer name (0.10)
    breakdown["personalisation"] = 0.10 if "marcus" in rl else 0.0

    score = _clamp(sum(breakdown.values()))

    parts = []
    labels = {
        "apology": "Apology/empathy",
        "timeline": "Timeline (7-10 days)",
        "policy_link": "Policy link",
        "professional_tone": "Professional tone",
        "personalisation": "Personalisation (name)",
    }
    for key, label in labels.items():
        val = breakdown[key]
        max_val = {"apology": 0.25, "timeline": 0.25, "policy_link": 0.25,
                   "professional_tone": 0.15, "personalisation": 0.10}[key]
        parts.append(f"{label}: {val:.2f}/{max_val:.2f}")

    feedback = "Breakdown — " + " | ".join(parts)
    if not response:
        feedback = "No draft_response provided."

    return TriageReward(score=score, feedback=feedback, breakdown=breakdown)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnv:
    """
    Email Triage reinforcement-learning environment.

    API
    ---
    - reset(task_name)  → TriageObservation
    - step(action)      → (TriageObservation, TriageReward, done: bool, info: dict)
    - state()           → TriageObservation
    - close()           → None (no-op, provided for spec compliance)

    Each episode is single-step: the agent receives all emails, submits one
    action, receives the graded reward, and the episode ends.
    """

    def __init__(self, task_name: str = "easy-categorize"):
        if task_name not in DATASETS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Valid tasks: {list(DATASETS.keys())}"
            )
        self.task_name = task_name
        self.episode_id: str = str(uuid.uuid4())
        self.current_step: int = 0
        self.is_done: bool = False
        self._last_reward: Optional[float] = None

    # ------------------------------------------------------------------

    async def reset(self, task_name: Optional[str] = None) -> TriageObservation:
        """Initialise a fresh episode (optionally switching tasks)."""
        if task_name and task_name != self.task_name:
            if task_name not in DATASETS:
                raise ValueError(f"Unknown task '{task_name}'")
            self.task_name = task_name

        self.episode_id = str(uuid.uuid4())
        self.current_step = 0
        self.is_done = False
        self._last_reward = None

        data = DATASETS[self.task_name]
        return TriageObservation(
            task_level=self.task_name,
            instructions=data["instructions"],
            emails=data["emails"],
            step_number=0,
            done=False,
        )

    # ------------------------------------------------------------------

    async def step(
        self, action: TriageAction
    ) -> Tuple[TriageObservation, TriageReward, bool, Dict[str, Any]]:
        """Execute one agent action and return (obs, reward, done, info)."""
        if self.is_done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        self.current_step += 1
        data = DATASETS[self.task_name]

        # Grade the action
        if self.task_name == "easy-categorize":
            reward = _grade_easy(action, data["solution"])
        elif self.task_name == "medium-prioritize":
            reward = _grade_medium(action, data["solution"])
        elif self.task_name == "hard-draft-response":
            reward = _grade_hard(action)
        else:
            reward = TriageReward(score=_clamp(0.0), feedback="Unknown task.")

        self._last_reward = reward.score
        self.is_done = True  # Single-step episodes

        obs = TriageObservation(
            task_level=self.task_name,
            instructions=data["instructions"],
            emails=data["emails"],
            step_number=self.current_step,
            done=True,
            last_reward=reward.score,
        )

        info = {
            "episode_id": self.episode_id,
            "task": self.task_name,
            "step": self.current_step,
        }

        return obs, reward, self.is_done, info

    # ------------------------------------------------------------------

    def state(self) -> TriageObservation:
        """Return the current observation without advancing the episode."""
        data = DATASETS[self.task_name]
        return TriageObservation(
            task_level=self.task_name,
            instructions=data["instructions"],
            emails=data["emails"],
            step_number=self.current_step,
            done=self.is_done,
            last_reward=self._last_reward,
        )

    async def close(self) -> None:
        """No-op: provided for API compatibility."""
        pass