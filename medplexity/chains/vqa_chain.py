from medplexity.benchmarks.vqa_utils import VQAInput
from medplexity.chains.chain import Chain, ChainOutput
from medplexity.llms.llm import LLM
from medplexity.prompts.prompt import Prompt
from medplexity.prompts.vqa_prompt import VQAPromptTemplate


class VQAChain(Chain):
    """Chain to perform visual question answering, expects an image and a question as input."""

    def __init__(
        self,
        llm: LLM,
        prompt: Prompt = VQAPromptTemplate(),
    ):
        self.llm = llm
        self.prompt = prompt

    def __call__(self, vqa_input: VQAInput) -> ChainOutput:
        full_prompt = self.prompt.format(
            question=vqa_input.question,
        )

        answer = self.llm(instruction=full_prompt, image=vqa_input.image)

        return ChainOutput(
            output=answer,
            output_metadata=None,
        )
