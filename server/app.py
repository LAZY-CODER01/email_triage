"""
Email Triage Environment — FastAPI Server

Creates the OpenEnv-compliant HTTP + WebSocket application using
create_app() from openenv-core, which automatically exposes:

  POST /reset    → initialize episode
  POST /step     → execute action
  GET  /state    → current state
  GET  /         → health check
  WS   /ws       → real-time WebSocket session
  GET  /web      → optional web UI (set ENABLE_WEB_INTERFACE=true)

Action format:
  Agents should send their payload inside the Action's metadata dict:
  {
    "metadata": {
      "categories":     {"e1": "Technical"},   # easy-categorize
      "priorities":     {"e1": "High", ...},   # medium-prioritize
      "draft_response": "Dear Marcus, ..."     # hard-draft-response
    }
  }
"""

import sys
import os

# Ensure the project root (parent of server/) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation

try:
    from server.email_environment import EmailEnvironment
except ImportError:
    from email_environment import EmailEnvironment


# create_app accepts a **callable** (class or factory) that returns an
# Environment instance.  Passing the class itself means each WebSocket
# session gets a fresh, isolated EmailEnvironment instance.
app = create_app(
    EmailEnvironment,       # env factory — called per session
    Action,                 # action type for schema generation
    Observation,            # observation type for schema generation
    env_name="email-triage-env",
)


def main() -> None:
    """Entry point for `uvicorn server.app:app` or direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()