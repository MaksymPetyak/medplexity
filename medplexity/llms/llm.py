from abc import abstractmethod


class LLM:
    """Base class for LLMs."""

    def __call__(self, instruction: str, suffix: str = "") -> str:
        raise NotImplementedError("LLM call method is not implemented")
