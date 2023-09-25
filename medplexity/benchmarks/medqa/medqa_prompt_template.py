from typing import List
import os


from pydantic import BaseModel, RootModel


class MedQAPromptExample(BaseModel):
    question: str
    options: list[str]
    explanation: str
    answer: str


MedQAExamplesQuestions = RootModel[List[MedQAPromptExample]]

class AnswerWithExplanation(BaseModel):
    answer: str
    explanation: str


INDEX_TO_OPTION = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}

def format_options(options: List[str]) -> str:

    return " ".join(
        [f"{INDEX_TO_OPTION[i]} {option}" for i, option in enumerate(options)]
    )

def format_medqa_answer(options: List[str], answer: str) -> str:
    answer_idx = options.index(answer)

    return INDEX_TO_OPTION[answer_idx]

def build_example_question() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "medqa_examples.json")

    with open(json_path, "r") as f:
        examples_obj = MedQAExamplesQuestions.model_validate_json(f.read())

        return [
            "Question: {question}\n {options}\nOutput: {output}\n".format(
                question=example.question,
                options=format_options(example.options),
                output=AnswerWithExplanation(
                    answer=format_medqa_answer(example.options, example.answer),
                    explanation=example.explanation,
                ).model_dump_json(indent=None),
            )
            for example in examples_obj.root
        ]


class MedQAPromptTemplate:
    """Chain-of-thought prompt for MedQA. Returns a JSON with answer and an explanation for it. Adapted from https://arxiv.org/abs/2305.09617"""

    parser = AnswerWithExplanation

    PROMPT = """The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a JSON with the answer (give back only the letter) and an explanation for it.
{examples}

Question: {question}
{options}
Output: """

    def format(self, question: str, options: list[str]):
        return self.PROMPT.format(
            examples="\n".join(
                build_example_question()
            ),
            question=question,
            options=options,
        )
