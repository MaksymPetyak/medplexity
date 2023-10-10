import re

from pydantic import BaseModel

from medplexity.benchmarks.multiple_choice_utils import (
    MultipleChoiceInput,
    MultipleChoicePromptExample,
)
from medplexity.chains.chain import Chain, ChainOutput
from medplexity.llms.llm import LLM
from medplexity.prompts.multiple_choice_prompt import MultipleChoiceChainOfThoughtPrompt


class AnswerWithExplanation(BaseModel):
    answer: str
    explanation: str


class MultipleChoiceEvaluationChain(Chain):
    """Chain to evaluate LLMs against multiple-choice question benchmarks."""

    def __init__(
        self,
        llm: LLM,
        prompt: MultipleChoiceChainOfThoughtPrompt = MultipleChoiceChainOfThoughtPrompt(),
        save_prompt: bool = False,
        examples: list[MultipleChoicePromptExample] = None,
    ):
        self.llm = llm
        self.prompt = prompt
        self.save_prompt = save_prompt
        self.examples = examples

    def __call__(self, input: MultipleChoiceInput) -> ChainOutput:
        examples = self.examples
        if examples is not None:
            examples = [
                self.prompt.format_example(
                    context=example.context,
                    question=example.question,
                    options=example.options,
                    answer=example.answer,
                    explanation=example.explanation,
                )
                for example in self.examples
            ]
            examples = "\n".join(examples)

        completed_prompt = self.prompt.format(
            question=input.question,
            options=input.options,
            context=input.context if input.context is not None else "",
            examples=examples if self.examples is not None else "",
        )

        output = self.llm(completed_prompt)

        explanation, answer = self._extract_explanation_and_answer(output)

        return ChainOutput(
            output=answer,
            output_metadata={
                "explanation": explanation,
                "prompt": completed_prompt if self.save_prompt else None,
            },
        )

    def _extract_explanation_and_answer(self, completion: str):
        explanation_match = re.search(
            r"Explanation: (.*?)\s+Answer: \((\w)\)", completion
        )

        if explanation_match:
            explanation = explanation_match.group(1)
            answer = explanation_match.group(2)
            return explanation, f"({answer})"
        else:
            raise ValueError(
                f"Could not extract explanation and answer from completion: {completion}"
            )
