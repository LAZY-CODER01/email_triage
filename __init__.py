"""
Email Triage Environment — Package Exports

Exposes the public API surface for use as a client library:
  from email_triage_env import TriageAction, TriageObservation, EmailTriageEnv
"""

from models import Email, TriageAction, TriageObservation, TriageReward
from env import EmailTriageEnv

__all__ = [
    "Email",
    "TriageAction",
    "TriageObservation",
    "TriageReward",
    "EmailTriageEnv",
]
