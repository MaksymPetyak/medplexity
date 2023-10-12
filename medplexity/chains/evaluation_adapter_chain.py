from typing import Optional, Callable, Any

from medplexity.chains.chain import Chain
from medplexity.llms.llm import LLM


class EvaluationAdapterChain(Chain):
    """Transforms a datapoint to the text input to the model and also adapts the output JSON"""

    def __init__(
        self,
        llm: LLM = None,
        chain: Chain = None,
        input_adapter: Optional[Callable[[Any], str]] = None,
        output_adapter: Optional[Callable[[Any], Any]] = None,
    ):
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

        self.llm = llm
        self.chain = chain

        if not self.llm and not self.chain:
            raise ValueError("Either an LLM or a chain must be provided")
        if self.llm and self.chain:
            raise ValueError("Only one of LLM or chain must be provided")

    def __call__(self, input: Any):
        if self.input_adapter:
            input = self.input_adapter(input)

        if self.chain:
            output = self.chain(input)
        if self.llm:
            output = self.llm(input)

        if self.output_adapter:
            output = self.output_adapter(output)

        return output
