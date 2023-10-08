from typing import List
import os


from pydantic import BaseModel, RootModel

from medplexity.benchmarks.multiple_choice_utils import AnswerWithExplanation
from medplexity.prompts.prompt import Prompt


class PubmedQAPromptExample(BaseModel):
    question: str
    context: str
    explanation: str
    answer: str


PumbedQAExamplesQuestions = RootModel[List[PubmedQAPromptExample]]

# Formulate the questions as multiple-choice still
ANSWER_OPTIONS = "(A) Yes (B) No (C) Maybe"
PUBMEDQA_ANSWER_TO_OPTION = {
    "yes": "(A)",
    "no": "(B)",
    "maybe": "(C)",
}


def format_options(options: List[str]) -> str:
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    return "\n".join(
        [f"({index_to_letter[i]}) {option}" for i, option in enumerate(options)]
    )


def build_example_question() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "pubmedqa_examples.json")

    with open(json_path, "r") as f:
        examples_obj = PumbedQAExamplesQuestions.model_validate_json(f.read())

        return [
            "Context: {context}\nQuestion: {question}. {options}\nOutput: {output}\n".format(
                context=example.context,
                question=example.question,
                options=ANSWER_OPTIONS,
                output=AnswerWithExplanation(
                    answer=PUBMEDQA_ANSWER_TO_OPTION[example.answer.lower()],
                    explanation=example.explanation,
                ).model_dump_json(indent=None),
            )
            for example in examples_obj.root
        ]


class PubmedQAPromptTemplate(Prompt):
    """Chain-of-thought prompt for PubmedQA. Returns a JSON with answer and an explanation for it. Adapted from https://arxiv.org/abs/2305.09617"""

    parser = AnswerWithExplanation

    PROMPT = """The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a JSON with the answer (give back only the letter) and an explanation for it.
{examples}

Context: {context}
Question: {question}. {options}
Output: """

    def format(self, question: str, context: list[str]):
        return self.PROMPT.format(
            examples="\n".join(build_example_question()),
            context=context,
            question=question,
            options=ANSWER_OPTIONS,
        )
