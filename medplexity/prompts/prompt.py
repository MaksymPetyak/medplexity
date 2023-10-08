import abc


class Prompt(abc.ABC):
    """Abstract class for prompts"""

    PROMPT = ""

    @abc.abstractmethod
    def format(self, *args, **kwargs) -> str:
        """Format the prompt with the given arguments"""
        pass
