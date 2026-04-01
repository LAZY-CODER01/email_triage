import asyncio
import os
import json
from openai import OpenAI
from env import EmailTriageEnv
from models import TriageAction
API_KEY = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY"))
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
TASK_NAME = os.getenv("TASK_NAME", "easy-categorize")
BENCHMARK = "email-triage-env"
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rews = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rews}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EmailTriageEnv(task_name=TASK_NAME)
    
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    
    obs = await env.reset()
    schema_str = json.dumps(TriageAction.model_json_schema())
    
    sys_prompt = """You are a customer support agent. 
You MUST return a valid JSON object. 
Do NOT wrap your output in a 'properties' key. Return the keys directly at the top level exactly like this:
{
  "categories": null,
  "priorities": null,
  "draft_response": "Your drafted email response goes here..."
}
"""
    
    user_prompt = f"Instructions: {obs.instructions}\nEmails: {[e.model_dump() for e in obs.emails]}"
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"}
    )
    
    action_data = json.loads(response.choices[0].message.content)
    action = TriageAction(**action_data)
    
    _, reward, done, _ = await env.step(action)
    
    log_step(1, json.dumps(action_data).replace('"', "'"), reward.score, done)
    log_end(reward.score >= 0.5, 1, reward.score, [reward.score])
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())