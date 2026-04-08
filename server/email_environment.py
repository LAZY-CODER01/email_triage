"""
Email Triage Environment — OpenEnv-compliant Server-Side Implementation

Wraps EmailTriageEnv in the openenv.core.env_server.interfaces.Environment base
class so that create_app() can expose it over HTTP + WebSocket.

The Environment abstract class requires:
  - reset(**kwargs)       → Observation
  - step(action)          → Observation
  - state (property)      → State
"""

from __future__ import annotations

import asyncio
import sys
import os
import uuid
from typing import Any, Dict, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from env import EmailTriageEnv, DATASETS
from models import TriageAction as _TriageAction


def _as_observation(triage_obs, reward: float = 0.0, done: bool = False) -> Observation:
    """Convert our TriageObservation → openenv Observation."""
    return Observation(
        done=done,
        reward=reward,
        metadata={
            "task_level": triage_obs.task_level,
            "instructions": triage_obs.instructions,
            "emails": [e.model_dump() for e in triage_obs.emails],
            "step_number": triage_obs.step_number,
            "last_reward": triage_obs.last_reward,
        },
    )


def _parse_action(action: Action) -> _TriageAction:
    """
    Extract a TriageAction from an openenv Action.

    The agent sends its response via the Action's metadata dict:
    {
      "categories":     {...},  # for easy-categorize
      "priorities":     {...},  # for medium-prioritize
      "draft_response": "...",  # for hard-draft-response
    }
    """
    meta = getattr(action, "metadata", {}) or {}
    try:
        return _TriageAction(
            categories=meta.get("categories"),
            priorities=meta.get("priorities"),
            draft_response=meta.get("draft_response"),
        )
    except Exception:
        return _TriageAction()


class EmailEnvironment(Environment):
    """
    OpenEnv-compliant Email Triage Environment.

    Agents select a task via reset(task_name=<task>) and submit their
    work in Action.metadata.

    Concurrent sessions: disabled by default (single env instance).
    Set SUPPORTS_CONCURRENT_SESSIONS = True if deploying with per-session
    factory via create_app.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True  # create_app instantiates one env per session

    def __init__(self) -> None:
        super().__init__()
        self._env = EmailTriageEnv(task_name="easy-categorize")
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a given task."""
        # Resolve task name from kwargs fallback
        task = task_name or kwargs.get("task_name") or self._env.task_name
        if task not in DATASETS:
            task = "easy-categorize"

        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )

        loop = asyncio.new_event_loop()
        try:
            triage_obs = loop.run_until_complete(self._env.reset(task_name=task))
        finally:
            loop.close()

        return _as_observation(triage_obs, reward=0.01, done=False)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one step. Agent payload goes in action.metadata."""
        self._state.step_count += 1
        triage_action = _parse_action(action)

        loop = asyncio.new_event_loop()
        try:
            triage_obs, reward_obj, done, _info = loop.run_until_complete(
                self._env.step(triage_action)
            )
        finally:
            loop.close()

        return _as_observation(triage_obs, reward=reward_obj.score, done=done)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """True async step used by the WebSocket handler."""
        self._state.step_count += 1
        triage_action = _parse_action(action)
        triage_obs, reward_obj, done, _info = await self._env.step(triage_action)
        return _as_observation(triage_obs, reward=reward_obj.score, done=done)

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """True async reset."""
        task = task_name or kwargs.get("task_name") or self._env.task_name
        if task not in DATASETS:
            task = "easy-categorize"
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        triage_obs = await self._env.reset(task_name=task)
        return _as_observation(triage_obs, reward=0.01, done=False)

    @property
    def state(self) -> State:
        return self._state
