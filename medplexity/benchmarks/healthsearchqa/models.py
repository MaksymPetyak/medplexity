from pydantic import BaseModel


class HealthSearchQAQuestion(BaseModel):
    # Some entries in the dataset are empty
    id: int | None
    question: str | None
