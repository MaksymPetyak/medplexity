from medplexity.chains.chain import Chain, ChainOutput
from medplexity.llms.llm import LLM
from medplexity.prompts.conversation_summarisation_prompt_template import (
    ConversationSummarisationPromptTemplate,
)


class SummarisationChain(Chain):
    """Chain to summarise given text or conversation"""

    def __init__(
        self,
        llm: LLM,
        prompt: ConversationSummarisationPromptTemplate = ConversationSummarisationPromptTemplate(),
        save_prompt: bool = False,
    ):
        self.llm = llm
        self.prompt = prompt
        self.save_prompt = save_prompt

    def __call__(self, text: str) -> ChainOutput:
        full_prompt = self.prompt.format(text)

        summary = self.llm(full_prompt)

        return ChainOutput(
            output=summary,
            output_metadata=None,
        )
