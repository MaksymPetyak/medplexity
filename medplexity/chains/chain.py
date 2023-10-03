from typing import Any

from pydantic import BaseModel


# TODO: add Generic type to ChainOutput
class ChainOutput(BaseModel):
    # Output we can compare against in the evaluation
    output: Any
    # Support passing additional information to explore the model's inner workings
    output_metadata: Any


class Chain:
    """Chains are used in conjunction with LLM and help to preprocess inputs and outputs for evaluation."""

    def __call__(self, *args, **kwargs) -> str | ChainOutput:
        raise NotImplementedError("LLM call method is not implemented")
