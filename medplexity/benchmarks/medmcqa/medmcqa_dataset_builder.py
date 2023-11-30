from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.medmcqa.models import MedMCQAQuestion
from medplexity.benchmarks.multiple_choice_utils import (
    MultipleChoiceInput,
    INDEX_TO_OPTION,
)
from medplexity.datasets.dataset import DataPoint, Dataset


class MedMCQAOutputMetadata(BaseModel):
    explanation: str | None
    subject_name: str


class MedMCQADataPoint(DataPoint):
    input: MultipleChoiceInput
    expected_output: str
    metadata: MedMCQAOutputMetadata


MedMCQADatasetSplitType = Literal["train", "validation", "test"]


class MedMCQADatasetBuilder(DatasetBuilder):
    """Multiple-choice questions designed to address real-world medical entrance exam questions like AIIMS & NEET PG.
    This dataset encompasses over 194k high-quality MCQs spanning 2.4k healthcare topics and 21 medical subjects. Questions are accompanied by an explanation of the correct answer.

    Paper: MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering

    2022 · Ankit Pal, Logesh Kumar Umapathi, Malaikannan Sankarasubbu
    <https://arxiv.org/abs/2203.14371>

    Train/validation/test splits available.

    Dataset version used: <https://huggingface.co/datasets/medmcqa>
    """

    EXAMPLE_QUESTIONS_PATH = Path(__file__).resolve().parent / "examples.json"

    def build_dataset(
        self, split_type: MedMCQADatasetSplitType, config=None
    ) -> Dataset[MedMCQADataPoint]:
        dataset = self.loader.load("medmcqa", split=split_type)

        dataset = [MedMCQAQuestion(**row) for row in dataset]

        data_points = [
            MedMCQADataPoint(
                id=question.id,
                input=MultipleChoiceInput(
                    question=question.question,
                    options=[question.opa, question.opb, question.opc, question.opd],
                ),
                expected_output=INDEX_TO_OPTION[question.cop],
                metadata=MedMCQAOutputMetadata(
                    explanation=question.exp,
                    subject_name=question.subject_name,
                ),
            )
            for question in dataset
            if question.cop is not None and question.cop != -1
        ]

        return Dataset[MedMCQADataPoint](
            data_points=data_points, description=self.__doc__
        )
