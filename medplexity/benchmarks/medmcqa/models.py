from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ChoiceType(str, Enum):
    single = "single"
    multi = "multi"


class MedMCQAQuestion(BaseModel):
    id: str
    question: str
    opa: str
    opb: str
    opc: str
    opd: str
    # Correct option
    cop: Optional[int]
    choice_type: ChoiceType
    # Explanation
    exp: Optional[str] = None
    subject_name: str
    topic_name: Optional[str] = None
