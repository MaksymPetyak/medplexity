from enum import Enum
from pathlib import Path
from typing import Literal

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.mmlu.models import MMLUQuestion
from medplexity.benchmarks.multiple_choice_utils import MultipleChoiceInput
from medplexity.datasets.dataset import DataPoint, Dataset


class MMLUSubsetConfig(str, Enum):
    clinical_knowledge = "clinical_knowledge"
    medical_genetics = "medical_genetics"
    anatomy = "anatomy"
    professional_medicine = "professional_medicine"
    college_biology = "college_biology"
    college_medicine = "college_medicine"


MMLUQADatasetSplitType = Literal["train", "validation", "test"]


class MMLUDataPoint(DataPoint):
    input: MultipleChoiceInput
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

    Paper: MMLU: Massive Multitask Language Understanding

    7 Sep 2020  ·  Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
    <https://arxiv.org/abs/2009.03300>

    We use the version on the hugging face datasets: <https://huggingface.co/datasets/lukaemon/mmlu>
    """

    EXAMPLE_QUESTIONS_PATH = Path(__file__).resolve().parent / "examples.json"

    def build_dataset(
        self,
        split_type: MMLUQADatasetSplitType = "train",
        config=None,
    ) -> Dataset[MMLUDataPoint]:
        if config is None:
            config = {"subset": MMLUSubsetConfig.clinical_knowledge}

        dataset = self.loader.load("lukaemon/mmlu", config["subset"], split=split_type)

        questions = [MMLUQuestion(**row) for row in dataset]

        data_points = [
            MMLUDataPoint(
                id=f"{split_type}-{i}",
                input=MultipleChoiceInput(
                    question=question.input,
                    options=[question.A, question.B, question.C, question.D],
                ),
                expected_output=f"({question.target})",
                metadata=None,
            )
            for i, question in enumerate(questions)
        ]

        return Dataset[MMLUDataPoint](data_points=data_points, description=self.__doc__)
