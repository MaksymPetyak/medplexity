from typing import Literal

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.medqa.models import MedQAQuestion
from pydantic import BaseModel

from medplexity.benchmarks.multiple_choice_utils import format_answer_to_letter
from medplexity.datasets.dataset import DataPoint, Dataset

from datasets import load_dataset


# TODO: extend and experiment with more types
MedQASubsetConfig = Literal["med_qa_en_bigbio_qa"]

MedQADatasetSplitType = Literal["train", "validation", "test"]


class MedQAInput(BaseModel):
    question: str
    options: list[str]


class MedQADataPoint(DataPoint):
    input: MedQAInput
    expected_output: str


class MedQADatasetBuilder(DatasetBuilder):
    """Multiple-choice questions based on the United States Medical License Exams (USMLE).

    Original paper: What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams

    28 Sep 2020 · Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits
    <https://arxiv.org/abs/2009.13081>

    Currently, uses only med_qa_en_bigbio_qa subset of the dataset, but can be extended to other subsets.

    Train/validation/test splits available.

    We use the following version uploaded on HuggingFace datasets: <https://huggingface.co/datasets/bigbio/med_qa>
    """

    def build_dataset(
        self,
        config_type: MedQASubsetConfig = "med_qa_en_bigbio_qa",
        split_type: MedQADatasetSplitType = "train",
        convert_answer_to_multiple_choice: bool = True,
    ) -> Dataset[MedQADataPoint]:
        dataset = load_dataset("bigbio/med_qa", config_type, split=split_type)

        questions = [MedQAQuestion(**row) for row in dataset]

        data_points = [
            MedQADataPoint(
                input=MedQAInput(
                    question=question.question,
                    options=question.choices,
                ),
                # always expect just one answer
                expected_output=format_answer_to_letter(
                    question.choices, question.answer[0]
                )
                if convert_answer_to_multiple_choice
                else question.answer[0],
                metadata=None,
            )
            for question in questions
        ]

        return Dataset[MedQADataPoint](
            data_points=data_points, description=self.__doc__
        )
