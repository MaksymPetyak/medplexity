import os

from medplexity.benchmarks.multiple_choice_utils import (
    build_example_questions,
    AnswerWithExplanation,
    format_options,
)
from medplexity.prompts.prompt import Prompt


def load_questions_from_file() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "mmlu_examples.json")

    return "\n".join(build_example_questions(json_path))


class MMLUPromptTemplate(Prompt):
    """Chain-of-thought prompt for MMLU. Returns a JSON with answer and an explanation for it. Adapted from https://arxiv.org/abs/2305.09617"""

    parser = AnswerWithExplanation

    PROMPT = """The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a JSON with the answer (give back only the letter) and an explanation for it.
{examples}

Question: {question}
{options}
Output: """

    def format(self, question: str, options: list[str]):
        return self.PROMPT.format(
            examples=load_questions_from_file(),
            question=question,
            options=format_options(options),
        )
