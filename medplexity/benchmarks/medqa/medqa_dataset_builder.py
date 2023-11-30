from enum import Enum
from pathlib import Path
from typing import Literal

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.medqa.models import MedQAQuestion

from medplexity.benchmarks.multiple_choice_utils import (
    format_answer_to_letter,
    MultipleChoiceInput,
)
from medplexity.datasets.dataset import DataPoint, Dataset


# TODO: extend and experiment with more types
class MedQASubsetConfig(str, Enum):
    med_qa_en_bigbio_qa = "med_qa_en_bigbio_qa"


MedQADatasetSplitType = Literal["train", "validation", "test"]


class MedQADataPoint(DataPoint):
    input: MultipleChoiceInput
    expected_output: str


class MedQADatasetBuilder(DatasetBuilder):
    """Multiple-choice questions based on the United States Medical License Exams (USMLE).

    Paper: What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams

    28 Sep 2020 · Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits
    <https://arxiv.org/abs/2009.13081>

    Currently, uses only med_qa_en_bigbio_qa subset of the dataset, but can be extended to other subsets.

    Train/validation/test splits available.

    We use the following version uploaded on HuggingFace datasets: <https://huggingface.co/datasets/bigbio/med_qa>
    """

    EXAMPLE_QUESTIONS_PATH = Path(__file__).resolve().parent / "examples.json"

    def build_dataset(
        self,
        split_type: MedQADatasetSplitType = "train",
        config=None,
    ) -> Dataset[MedQADataPoint]:
        if config is None:
            config = {"subset": MedQASubsetConfig.med_qa_en_bigbio_qa}

        dataset = self.loader.load("bigbio/med_qa", config["subset"], split=split_type)

        questions = [MedQAQuestion(**row) for row in dataset]

        data_points = [
            MedQADataPoint(
                id=question.id,
                input=MultipleChoiceInput(
                    question=question.question,
                    options=question.choices,
                ),
                # always expect just one answer
                expected_output=format_answer_to_letter(
                    question.choices, question.answer[0]
                ),
                metadata=None,
            )
            for question in questions
        ]

        return Dataset[MedQADataPoint](
            data_points=data_points, description=self.__doc__
        )
