from enum import Enum

from medplexity.chains.chain import Chain, ChainOutput

from langchain.chains.base import Chain as LangchainChain


class LangChainOutputDictKeys(str, Enum):
    input: str = "input"
    output: str = "output"
    intermediate_steps: str = "intermediate_steps"


class LangchainWrapper(Chain):
    """This wrapper simplifies the invocation of the LangchainChain, and provides an option
    to store intermediate steps in the output."""

    def __init__(
        self, langchain_chain: LangchainChain, store_intermediate_steps: bool = False
    ):
        self.chain = langchain_chain
        self.store_intermediate_steps = store_intermediate_steps

    def __call__(self, input: str, **kwargs) -> ChainOutput:
        response = self.chain.invoke(
            input={"input": input},
            **kwargs,
        )
        output_metadata = (
            None
            if not self.store_intermediate_steps
            else {
                LangChainOutputDictKeys.intermediate_steps: response[
                    LangChainOutputDictKeys.intermediate_steps
                ],
            }
        )

        output = response[LangChainOutputDictKeys.output]

        return ChainOutput(
            output=output,
            output_metadata=output_metadata,
        )
