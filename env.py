import asyncio
from models import TriageObservation, TriageAction, TriageReward, Email

class EmailTriageEnv:
    def __init__(self, task_name: str = "easy-categorize"):
        self.task_name = task_name
        self.current_step = 0
        self.max_steps = 1
        self.is_done = False
        
        # Internal state based on task
        self.dataset = {
            "easy-categorize": {
                "emails": [Email(id="e1", subject="Cannot login", body="I keep getting a 401 error when I try to access my dashboard.")],
                "instructions": "Categorize the email. Valid categories: Billing, Technical, Sales.",
                "solution": {"e1": "Technical"}
            },
            "medium-prioritize": {
                "emails": [
                    Email(id="e1", subject="Question about enterprise plan", body="We are considering upgrading next quarter."),
                    Email(id="e2", subject="URGENT: Production down", body="All our APIs are returning 500s! Help!"),
                    Email(id="e3", subject="Invoice #442", body="Can you send me a PDF of last month's invoice?")
                ],
                "instructions": "Assign priorities. Valid priorities: High, Medium, Low.",
                "solution": {"e1": "Medium", "e2": "High", "e3": "Low"}
            },
            "hard-draft-response": {
                "emails": [Email(id="e1", subject="Angry: Missing Refund", body="I cancelled 5 days ago and haven't seen my money! This is theft! Refund me now!")],
                "instructions": "Draft a response. You must apologize, assure them it takes 7-10 business days, and include the link: 'docs.company.com/refunds'.",
                "solution": {} # Graded via heuristics
            }
        }

    async def reset(self):
        self.current_step = 0
        self.is_done = False
        task_data = self.dataset[self.task_name]
        return TriageObservation(
            task_level=self.task_name,
            instructions=task_data["instructions"],
            emails=task_data["emails"]
        )

    async def step(self, action: TriageAction):
        if self.is_done:
            raise ValueError("Episode is already done. Call reset().")
        
        self.current_step += 1
        reward_score = 0.0
        feedback = ""

        # Graders
        if self.task_name == "easy-categorize":
            expected = self.dataset["easy-categorize"]["solution"]
            if action.categories and action.categories.get("e1") == expected["e1"]:
                reward_score = 1.0
                feedback = "Correctly categorized."
            else:
                feedback = "Incorrect or missing category."

        elif self.task_name == "medium-prioritize":
            expected = self.dataset["medium-prioritize"]["solution"]
            correct = 0
            if action.priorities:
                for k, v in expected.items():
                    if action.priorities.get(k) == v:
                        correct += 1
            reward_score = correct / 3.0
            feedback = f"Correctly prioritized {correct}/3 emails."

        elif self.task_name == "hard-draft-response":
            response = action.draft_response or ""
            score = 0.0
            if "sorry" in response.lower() or "apologize" in response.lower():
                score += 0.33
            if "7-10" in response or "7 to 10" in response:
                score += 0.33
            if "docs.company.com/refunds" in response:
                score += 0.34
            reward_score = min(score, 1.0)
            feedback = f"Response graded based on policy adherence: {reward_score:.2f}/1.0"

        self.is_done = True
        return self.state(), TriageReward(score=reward_score, feedback=feedback), self.is_done, {}

    def state(self):
        task_data = self.dataset[self.task_name]
        return TriageObservation(
            task_level=self.task_name,
            instructions=task_data["instructions"],
            emails=task_data["emails"]
        )

    async def close(self):
        pass