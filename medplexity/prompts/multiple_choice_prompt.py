import re
import string
from typing import List

from pydantic import BaseModel

from medplexity.prompts.prompt import Prompt


class AnswerWithExplanation(BaseModel):
    answer: str
    explanation: str


class MultipleChoiceChainOfThoughtPrompt(Prompt):
    """Prompt for multiple-choice questions, adapted from https://arxiv.org/abs/2305.09617"""

    QUESTION_PROMPT = """{context}
Question: {question}
{options}
Explanation: {explanation}
Answer: {answer}"""

    PROMPT = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion,
starting by summarizing the available information. Output a single option from the given options as the final answer.
{examples}{answer_template}
{context}
Question: {question}
{options}
"""

    # If there are no examples provided we additionally provide a template for the answer
    ANSWER_TEMPLATE = """Use the following format for your answer:
Explanation: Reasoning for your answer
Answer: (A) | (B) | (C), etc. Give only the option and nothing else.
"""

    def format(
        self, question: str, options: list[str], context: str = "", examples: str = ""
    ):
        return self.PROMPT.format(
            question=question,
            options=self.format_options(options),
            context=context,
            examples=examples,
            answer_template=self.ANSWER_TEMPLATE if examples == "" else "",
        )

    def format_example(
        self,
        context: str,
        question: str,
        options: list[str],
        answer: str,
        explanation: str = "",
    ):
        return self.QUESTION_PROMPT.format(
            context=context,
            question=question,
            options=self.format_options(options),
            explanation=explanation,
            answer=answer,
        )

    def format_options(self, options: List[str]):
        """Uses letter format "(A)" for formatting options for questions."""

        if len(options) > len(string.ascii_uppercase):
            raise ValueError("Too many options for standard multiple-choice formatting")

        formatted_options = [
            f"({string.ascii_uppercase[i]}) {option}"
            for i, option in enumerate(options)
        ]

        return " ".join(formatted_options)

    @staticmethod
    def extract_explanation_and_answer(completion: str) -> AnswerWithExplanation:
        """Helper function to parse the expected completion format and extract the explanation and answer."""
        explanation_match = re.search(
            r"Explanation: (.*?)\s+Answer: \((\w)\)", completion
        )

        if explanation_match:
            explanation = explanation_match.group(1)
            answer = explanation_match.group(2)
            return AnswerWithExplanation(
                answer=f"({answer})",
                explanation=explanation,
            )
        else:
            raise ValueError(
                f"Could not extract explanation and answer from completion: {completion}"
            )
