from enum import Enum
from typing import Literal

from pydantic import BaseModel

from benchmarks.dataset_builder import DatasetBuilder
from benchmarks.mmlu.models import MMLUQuestion
from medplexity.datasets.dataset import DataPoint, Dataset

from datasets import load_dataset


class MMLUSubsetConfig(str, Enum):
    clinical_knowledge = "clinical_knowledge"
    medical_genetics = "medical_genetics"
    anatomy = "anatomy"
    professional_medicine = "professional_medicine"
    college_biology = "college_biology"
    college_medicine = "college_medicine"


MMLUQADatasetSplitType = Literal["train", "validation", "test"]


class MMLUInput(BaseModel):
    question: str
    options: list[str]


class MMLUDataPoint(DataPoint):
    input: MMLUInput
    expected_output: str


class MMLUDatasetBuilder(DatasetBuilder):
    """MMLU (Massive Multitask Language Understanding) is a massive multitask test consisting of multiple-choice questions from various domains.

    We are interested in the tasks that could be related to the medical domain, so use the following subsets:
    - Clinical knowledge
    - Medical genetics
    - Anatomy
    - Professional medicine
    - College biology
    - College medicine

    Original paper: MMLU: Massive Multitask Language Understanding

    7 Sep 2020  ·  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
    <https://arxiv.org/abs/2009.03300>

    We use the version on the hugging face datasets: <https://huggingface.co/datasets/lukaemon/mmlu>
    """

    def build_dataset(
        self,
        config_type: MMLUSubsetConfig = MMLUSubsetConfig.clinical_knowledge,
        split_type: MMLUQADatasetSplitType = "train",
    ) -> Dataset[MMLUDataPoint]:
        dataset = load_dataset("lukaemon/mmlu", config_type, split=split_type)

        questions = [MMLUQuestion(**row) for row in dataset]

        data_points = [
            MMLUDataPoint(
                input=MMLUInput(
                    question=question.input,
                    options=[question.A, question.B, question.C, question.D],
                ),
                expected_output=f"({question.target})",
                metadata=None,
            )
            for question in questions
        ]

        return Dataset[MMLUDataPoint](data_points=data_points, description=self.__doc__)
