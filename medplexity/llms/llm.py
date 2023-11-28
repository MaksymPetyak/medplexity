import abc
from typing import Optional

from PIL.Image import Image
from dotenv import load_dotenv

load_dotenv()


class LLM(abc.ABC):
    """Base class for LLMs."""

    model: str

    def __call__(
        self, instruction: str, suffix: str = "", image: Optional[Image] = None
    ) -> str:
        raise NotImplementedError("LLM call method is not implemented")
