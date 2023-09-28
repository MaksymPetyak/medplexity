from pydantic import BaseModel
from typing import List


class MedQAQuestion(BaseModel):
    id: str
    question_id: str
    document_id: str
    question: str
    type: str
    choices: List[str]
    context: str
    # Should contain just one item
    answer: List[str]
