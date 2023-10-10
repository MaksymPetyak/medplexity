from medplexity.chains.chain import Chain, ChainOutput
from medplexity.llms.llm import LLM
from medplexity.prompts.prompt import Prompt


class LLMChain(Chain):
    """Simple chain to invoke an LLM on a given string input, optionally with a prompt accepting a single input."""

    def __init__(
        self,
        llm: LLM,
        prompt: Prompt | None = None,
        save_prompt: bool = False,
    ):
        self.llm = llm
        self.prompt = prompt
        self.save_prompt = save_prompt

    def __call__(self, input: str) -> ChainOutput:
        full_prompt = input

        if self.prompt:
            full_prompt = self.prompt.format(input)

        completion = self.llm(full_prompt)

        return ChainOutput(
            output=completion,
            output_metadata={
                "prompt": full_prompt if self.save_prompt else None,
            },
        )
