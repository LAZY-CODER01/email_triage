from pydantic import BaseModel
from typing import List, Dict, Optional

class Email(BaseModel):
    id: str
    subject: str
    body: str

class TriageObservation(BaseModel):
    task_level: str
    instructions: str
    emails: List[Email]

class TriageAction(BaseModel):
    categories: Optional[Dict[str, str]] = None 
    priorities: Optional[Dict[str, str]] = None  
    draft_response: Optional[str] = None    

class TriageReward(BaseModel):
    score: float
    feedback: str