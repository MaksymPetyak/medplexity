from enum import Enum

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.healthsearchqa.models import HealthSearchQAQuestion
from medplexity.datasets.dataset import Dataset, DataPoint


class HealthSearchQADataPoint(DataPoint):
    input: str
    expected_output: None
    metadata: None


class HealthSearchQASubsetConfig(str, Enum):
    all_data = "all_data"
    _140_question_subset = "140_question_subset"


class HealthSearchQADatasetBuilder(DatasetBuilder):
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
        split_type: str = "train",
        config=None,
    ) -> Dataset[HealthSearchQADataPoint]:
        if config is None:
            config = {"subset": HealthSearchQASubsetConfig.all_data}

        dataset = self.loader.load(
            "katielink/healthsearchqa", config["subset"], split=split_type
        )

        questions = [HealthSearchQAQuestion(**row) for row in dataset]

        data_points = [
            HealthSearchQADataPoint(
                id=str(question.id),
                input=question.question,
                expected_output=None,
                metadata=None,
            )
            for question in questions
            if question.id is not None and question.question is not None
        ]

        return Dataset[HealthSearchQADataPoint](
            data_points=data_points, description=self.__doc__
        )
