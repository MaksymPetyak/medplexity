from typing import Optional, Callable, Any

from medplexity.datasets.dataset import DataPoint
from medplexity.llms.llm import LLM


class EvaluationAdapterChain:
    """Transforms a datapoint to the text input to the model and also adapts the output JSON"""

    def __init__(
        self,
        llm: LLM,
        input_adapter: Optional[Callable[[DataPoint], str]] = None,
        output_adapter: Optional[Callable[[str], Any]] = None,
    ):
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

        self.llm = llm

    def __call__(self, input: Any):
        if self.input_adapter:
            input = self.input_adapter(input)

        output = self.llm(input)

        if self.output_adapter:
            output = self.output_adapter(output)

        return output
