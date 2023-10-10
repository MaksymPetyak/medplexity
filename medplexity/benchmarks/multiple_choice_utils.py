from pathlib import Path
from typing import List

from pydantic import BaseModel, RootModel


class MultipleChoicePromptExample(BaseModel):
    question: str
    options: list[str]
    explanation: str
    answer: str
    context: str | None = None


class MultipleChoiceInput(BaseModel):
    question: str
    options: list[str]
    context: str | None = None
    examples: str | None = None


MultipleChoiceExampleQuestions = RootModel[List[MultipleChoicePromptExample]]


INDEX_TO_OPTION = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}


def format_options(options: List[str]) -> str:
    return " ".join(
        [f"{INDEX_TO_OPTION[i]} {option}" for i, option in enumerate(options)]
    )


def format_answer_to_letter(options: List[str], answer: str) -> str:
    answer_idx = options.index(answer)

    return INDEX_TO_OPTION[answer_idx]


def load_example_questions_from_json(
    file_path: Path | str,
) -> list[MultipleChoicePromptExample]:
    with open(file_path, "r") as f:
        examples_wrapper = MultipleChoiceExampleQuestions.model_validate_json(f.read())

        examples = examples_wrapper.root
        for example in examples:
            example.answer = format_answer_to_letter(example.options, example.answer)

        return examples
