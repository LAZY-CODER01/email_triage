"""
Email Triage Environment — Typed Data Models

Observation: what the agent sees each step.
Action:      what the agent sends back.
Reward:      score + feedback for the trainer.
Email:       a single email object (id, subject, body, sender, etc.)
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """A single customer support email."""
    id: str = Field(..., description="Unique email ID within the episode (e.g. 'e1')")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")
    sender: str = Field(..., description="Sender's display name or email address")
    timestamp: str = Field(..., description="ISO-8601 timestamp when the email arrived")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation grouping")


# ---------------------------------------------------------------------------
# Observation — what the agent receives
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """
    Observation returned by reset() and step().

    Contains:
    - task_level: which task scenario is active
    - instructions: plain-text task description + constraints
    - emails: the list of email objects to act on
    - step_number: which step we are on (0 = initial)
    - done: True once the episode has ended
    """
    task_level: str = Field(..., description="Active task name")
    instructions: str = Field(..., description="Task instructions for the agent")
    emails: List[Email] = Field(..., description="Inbox emails for this episode")
    step_number: int = Field(0, description="Current step index")
    done: bool = Field(False, description="True once episode has concluded")
    last_reward: Optional[float] = Field(None, description="Reward from the previous step")


# ---------------------------------------------------------------------------
# Action — what the agent sends
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Action submitted by the agent.

    Exactly ONE of the three fields should be populated per task:
      - categories:     {email_id -> "Billing"|"Technical"|"Sales"|"General"}
      - priorities:     {email_id -> "High"|"Medium"|"Low"}
      - draft_response: full text of the drafted email reply
    """
    categories: Optional[Dict[str, str]] = Field(
        None,
        description="Map of email ID → category for easy-categorize task. "
                    "Valid values: Billing, Technical, Sales, General"
    )
    priorities: Optional[Dict[str, str]] = Field(
        None,
        description="Map of email ID → priority for medium-prioritize task. "
                    "Valid values: High, Medium, Low"
    )
    draft_response: Optional[str] = Field(
        None,
        description="Full drafted reply text for hard-draft-response task"
    )


# ---------------------------------------------------------------------------
# Reward — score + feedback returned alongside each observation
# ---------------------------------------------------------------------------

class TriageReward(BaseModel):
    """Scalar reward + human-readable feedback for the current step."""
    score: float = Field(..., gt=0.0, lt=1.0, description="Normalized score strictly in (0.0, 1.0)")
    feedback: str = Field(..., description="What the agent did well or missed")
    breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Per-criterion scores for hard tasks"
    )