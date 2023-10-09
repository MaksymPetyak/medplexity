from enum import Enum

from datasets import load_dataset
from pydantic import BaseModel

from medplexity.benchmarks.dataset_factory import DatasetFactory
from medplexity.benchmarks.healthsearchqa.models import HealthSearchQAQuestion
from medplexity.datasets.dataset import Dataset, DataPoint


class HealthSearchQAInput(BaseModel):
    question: str


class HealthSearchQADataPoint(DataPoint):
    input: HealthSearchQAInput
    expected_output: None
    metadata: None


class HealthSearchQASubsetConfig(str, Enum):
    all_data = "all_data"
    _140_question_subset = "140_question_subset"


class HealthSearchQADatasetFactory(DatasetFactory):
    """Dataset of consumer health questions released by Google for the Med-PaLM paper.
    This HealthSearchQA dataset consists of 3,173 commonly searched consumer health questions. These questions were curated using seed medical conditions and their associated symptoms, reflecting real-world consumer concerns in the healthcare domain.

    Paper: Large Language Models Encode Clinical Knowledge

    2022 * Singhal, K., Azizi, S., Tu, T. et al.
    <https://arxiv.org/abs/2212.13138>

    No dataset splitting (only "train" split).

    Dataset version used: <https://huggingface.co/datasets/katielink/healthsearchqa>
    """

    def build_dataset(
        self,
        subset: HealthSearchQASubsetConfig = HealthSearchQASubsetConfig.all_data,
    ) -> Dataset[HealthSearchQADataPoint]:
        # No splitting, so just set split='train'
        dataset = load_dataset("katielink/healthsearchqa", subset, split="train")

        questions = [HealthSearchQAQuestion(**row) for row in dataset]

        data_points = [
            HealthSearchQADataPoint(
                input=HealthSearchQAInput(
                    question=question.question,
                ),
                expected_output=None,
                metadata=None,
            )
            for question in questions
            if question.id is not None and question.question is not None
        ]

        return Dataset[HealthSearchQADataPoint](
            data_points=data_points, description=self.__doc__
        )
