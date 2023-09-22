from typing import List, Literal

from pydantic import BaseModel


PubmedQADecision = Literal["yes", "no", "maybe"]


class Context(BaseModel):
    contexts: List[str]
    labels: List[str]
    meshes: List[str]


class PubMedQAQuestion(BaseModel):
    question: str
    context: Context
    long_answer: str
    final_decision: PubmedQADecision
