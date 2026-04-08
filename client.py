"""
Email Triage Environment — Client-Side EnvClient

Provides typed access to a running Email Triage environment server.

Usage:
    import asyncio
    from client import EmailTriageClient
    from models import TriageAction

    async def main():
        async with EmailTriageClient(base_url="https://your-space.hf.space") as env:
            # Reset for a specific task
            obs = await env.reset_typed(task_name="medium-prioritize")
            print(obs.metadata["instructions"])

            # Submit an action via metadata dict (how the server expects it)
            result = await env.step_typed(
                TriageAction(priorities={"e1": "Medium", "e2": "High",
                                         "e3": "Low", "e4": "Low", "e5": "High"})
            )
            print("Score:", result.reward)

    asyncio.run(main())

Note on the Action format:
    The openenv-core Action base class uses a `metadata` dict to carry payload.
    EmailTriageClient.step_typed() wraps TriageAction fields into metadata automatically.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

from openenv.core.client import EnvClient
from openenv.core.env_server.types import Action, Observation

from models import TriageAction


class EmailTriageClient(EnvClient):
    """
    Async HTTP/WebSocket client for the Email Triage environment.

    Connects to a running instance (Docker container or HF Space) and
    provides typed helpers on top of the raw EnvClient interface.
    """

    async def reset_typed(
        self,
        task_name: str = "easy-categorize",
        episode_id: Optional[str] = None,
    ) -> Observation:
        """Reset the environment for the specified task."""
        return await self.reset(task_name=task_name, episode_id=episode_id)

    async def step_typed(self, action: TriageAction) -> Observation:
        """
        Execute a TriageAction against the running server.

        Wraps the TriageAction fields into Action.metadata so the server
        can deserialize them correctly.
        """
        # The server's EmailEnvironment reads payload from action.metadata
        openenv_action = Action(
            metadata=action.model_dump(exclude_none=True)
        )
        return await self.step(openenv_action)
