from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel

from medplexity.benchmarks.dataset_factory import DatasetFactory
from medplexity.datasets.dataset import DataPoint, Dataset


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


MedMCQADatasetSplitType = Literal["train", "validation", "test"]


class MedMCQADatasetFactory(DatasetFactory):
    """Multiple-choice questions designed to address real-world medical entrance exam questions like AIIMS & NEET PG.
    This dataset encompasses over 194k high-quality MCQs spanning 2.4k healthcare topics and 21 medical subjects. Questions are accompanied by an explanation of the correct answer.

    Original paper: MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering

    2022 · Ankit Pal, Logesh Kumar Umapathi, Malaikannan Sankarasubbu
    <https://arxiv.org/abs/2203.14371>

    Train/validation/test splits available.

    Dataset version used: <https://huggingface.co/datasets/medmcqa>
    """

    def build_dataset(
        self, split_type: MedMCQADatasetSplitType
    ) -> Dataset[MedMCQADataPoint]:
        questions = load_dataset("medmcqa", split=split_type)

        data_points = [
            MedMCQADataPoint(
                input=MedMCQAInput(
                    question=question.question,
                    options=[question.opa, question.opb, question.opc, question.opd],
                ),
                expected_output=question.cop,
                metadata=MedMCQAOutputMetadata(
                    explanation=question.exp,
                    subject_name=question.subject_name,
                ),
            )
            for question in questions
            if question.cop is not None
        ]

        return Dataset[MedMCQADataPoint](
            data_points=data_points, description=self.__doc__
        )