from pathlib import Path
from typing import List

from pydantic import BaseModel, RootModel


class MultipleChoicePromptExample(BaseModel):
    question: str
    options: list[str]
    explanation: str
    answer: str


MultipleChoiceExampleQuestions = RootModel[List[MultipleChoicePromptExample]]

class AnswerWithExplanation(BaseModel):
    answer: str
    explanation: str


INDEX_TO_OPTION = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}
def format_options(options: List[str]) -> str:

    return " ".join(
        [f"{INDEX_TO_OPTION[i]} {option}" for i, option in enumerate(options)]
    )

def format_answer_to_letter(options: List[str], answer: str) -> str:
    answer_idx = options.index(answer)

    return INDEX_TO_OPTION[answer_idx]

def load_example_questions_from_json(file_path: Path | str) -> MultipleChoiceExampleQuestions:
    with open(file_path, "r") as f:
        examples = MultipleChoiceExampleQuestions.model_validate_json(f.read())

        return examples

def build_example_questions(file_path: Path | str) -> List[str]:
    examples = load_example_questions_from_json(file_path)

    return [
        "Question: {question}\n {options}\nOutput: {output}\n".format(
            question=example.question,
            options=format_options(example.options),
            output=AnswerWithExplanation(
                answer=format_answer_to_letter(example.options, example.answer),
                explanation=example.explanation,
            ).model_dump_json(indent=None),
        )
        for example in examples.root
    ]


