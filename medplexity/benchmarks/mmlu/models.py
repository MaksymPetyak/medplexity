from pydantic import BaseModel


class MMLUQuestion(BaseModel):
    input: str
    A: str
    B: str
    C: str
    D: str
    target: str
