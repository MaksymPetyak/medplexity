from enum import Enum
from typing import List

from pydantic import BaseModel


class PubmedQADecision(str, Enum):
    yes = "yes"
    no = "no"
    maybe = "maybe"


class Context(BaseModel):
    contexts: List[str]
    labels: List[str]
    meshes: List[str]


class PubMedQAQuestion(BaseModel):
    question: str
    context: Context
    long_answer: str
    final_decision: PubmedQADecision
