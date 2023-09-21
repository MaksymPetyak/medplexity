from typing import List

from pydantic import BaseModel


class MedMCQAPromptExample(BaseModel):
    question: str
    options: list[str]
    explanation: str
    answer: str

class AnswerWithExplanation(BaseModel):
    answer: str
    explanation: str


EXAMPLES = [
    MedMCQAPromptExample(
        question="Maximum increase in prolactin level is caused by:",
        options=["Risperidone", "Clozapine", "Olanzapine", "Aripiprazole"],
        explanation="Let’s solve this step-by-step, referring to authoritative sources as needed. Clozapine generally does not raise prolactin levels. Atypicals such as olanzapine and aripiprazole cause small if no elevation. Risperidone is known to result in a sustained elevated prolactin level. Therefore risperidone is likely to cause the maximum increase in prolactin level.",
        answer="(A)",
    ),
    MedMCQAPromptExample(
        question="What is the age of routine screening mammography?",
        options=["20 years", "30 years", "40 years", "50 years"],
        explanation="Let’s solve this step-by-step, referring to authoritative sources as needed. The age of routine screening depends on the country you are interested in and varies widely. For the US, it is 40 years of age according to the American Cancer Society. In Europe, it is typically closer to 50 years. For a patient based in the US, the best answer is 40 years.",
        answer="(C)",
    )
]

def format_options(options: List[str]) -> str:
    index_to_letter = { 0: "A", 1: "B", 2: "C", 3: "D" }

    return "\n".join([f"({index_to_letter[i]}) {option}" for i, option in enumerate(options)])

def build_example_question(example: MedMCQAPromptExample):
    return f"""
    Question: {example.question}
    {format_options(example.options)}
    Output: {AnswerWithExplanation(answer=example.answer, explanation=example.explanation).model_dump_json(indent=None)}
    """


class MedMCQAPromptTemplate():
    """Chain-of-thought prompt for MedMCQA. Returns a JSON with answer and an explanation for it. Adapted from https://arxiv.org/abs/2305.09617"""
    parser = AnswerWithExplanation

    PROMPT = """The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output a JSON with the answer (give back only the letter) and an explanation for it.
{examples}
    
Question: {question}
{options}
Output: """

    def format(self, question: str, options: list[str]):
        return self.PROMPT.format(
            examples="\n".join([build_example_question(example) for example in EXAMPLES]),
            question=question,
            options=options,
        )