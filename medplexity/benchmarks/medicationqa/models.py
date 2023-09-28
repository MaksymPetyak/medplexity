from pydantic import BaseModel, Field


class MedicationQAEntry(BaseModel):
    question: str = Field(..., alias="Question")
    focus: str | None = Field(..., alias="Focus (Drug)")
    question_type: str | None = Field(..., alias="Question Type")
    answer: str | None = Field(..., alias="Answer")
    section_title: str | None = Field(..., alias="Section Title")
    url: str | None = Field(..., alias="URL")
