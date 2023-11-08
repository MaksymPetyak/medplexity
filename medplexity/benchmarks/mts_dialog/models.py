from pydantic import BaseModel


class MTSDialogEntry(BaseModel):
    ID: int
    section_header: str
    section_text: str
    dialogue: str
