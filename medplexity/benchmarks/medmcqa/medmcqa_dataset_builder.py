from pydantic import BaseModel

from medplexity.benchmarks.medmcqa.medmcqa_loader import MedMCQALoader, DATASET_TYPE
from medplexity.datasets.dataset import DataPoint


class MedMCQAInput(BaseModel):
    question: str
    options: list[str]


class MedMCQAOutputMetadata(BaseModel):
    explanation: str | None
    subject_name: str


class MedMCQADataPoint(DataPoint):
    input: MedMCQAInput
    expected_output: int
    metadata: MedMCQAOutputMetadata



class MedMCQADatasetBuilder():
    def __init__(self, loader: MedMCQALoader | None = None):
        self.loader = loader or MedMCQALoader()

    def build_dataset(self, dataset_type: DATASET_TYPE) -> list[MedMCQADataPoint]:
        questions = self.loader.load_questions(dataset_type)

        data_points = [
            MedMCQADataPoint(
                input=MedMCQAInput(
                    question=question.question,
                    options=[question.opa, question.opb, question.opc, question.opd]
                ),
                expected_output=question.cop,
                metadata=MedMCQAOutputMetadata(
                    explanation=question.exp,
                    subject_name=question.subject_name,
                )
            )
            for question in questions
            if question.cop is not None
        ]

        return data_points
