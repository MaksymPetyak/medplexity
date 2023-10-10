import string
from typing import List

from medplexity.prompts.prompt import Prompt


class MultipleChoiceChainOfThoughtPrompt(Prompt):
    """Prompt for multiple-choice questions, adapted from https://arxiv.org/abs/2305.09617"""

    QUESTION_PROMPT = """{context}
Question: {question}
{options}
Explanation: {explanation}
Answer: {answer}"""

    PROMPT = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion,
starting by summarizing the available information. Output a single option from the given options as the final answer.
{examples}

{context}
Question: {question}
{options}
"""

    def format(
        self, question: str, options: list[str], context: str = "", examples: str = ""
    ):
        return self.PROMPT.format(
            question=question,
            options=self.format_options(options),
            context=context,
            examples=examples,
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
        """Uses letter format "(A)" for formatting options fro questions."""

        if len(options) > len(string.ascii_uppercase):
            raise ValueError("Too many options for standard multiple-choice formatting")

        formatted_options = [
            f"({string.ascii_uppercase[i]}) {option}"
            for i, option in enumerate(options)
        ]

        return " ".join(formatted_options)
