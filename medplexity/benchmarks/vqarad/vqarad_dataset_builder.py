from typing import Literal

from PIL.Image import Image

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.vqa_utils import ImageBaseModel
from medplexity.benchmarks.vqarad.models import VQARadEntry
from medplexity.datasets.dataset import DataPoint, Dataset


VQARadSplitTypes = Literal["train", "test"]


class VQARadInput(ImageBaseModel):
    image: Image
    question: str


class VQARadMetadata(ImageBaseModel):
    expected_answer: str


class VQARadDataPoint(DataPoint):
    input: VQARadInput
    # We avoid comparison of open-ended answers so put the answer in the metadata.
    expected_output: None

    class Config:
        arbitrary_types_allowed = True


class VQARadDatasetBuilder(DatasetBuilder):
    """
    VQA-RAD (Visual Question Answering in Radiology) is a dataset comprising 3,515 question-answer pairs on 315 radiology images, designed for visual question answering tasks.

    Train/test splits available.

    Paper: "A dataset of clinically generated visual questions and answers about radiology images"
     · 2018 · Jason J. Lau, Soumya Gayen, Asma Ben Abacha, Dina Demner-Fushman
    <https://www.nature.com/articles/sdata2018251>

    Dataset version used: <https://huggingface.co/datasets/flaviagiammarino/vqa-rad>
    """

    def build_dataset(
        self,
        split_type: VQARadSplitTypes = "test",
        config=None,
    ) -> Dataset[VQARadDataPoint]:
        vqa_raw_data = self.loader.load("flaviagiammarino/vqa-rad", split=split_type)

        entries = [VQARadEntry(**row) for row in vqa_raw_data]

        data_points = [
            VQARadDataPoint(
                id=f"{split_type}-{i}",
                input=VQARadInput(
                    image=entry.image,
                    question=entry.question,
                ),
                expected_output=None,
                metadata=VQARadMetadata(
                    expected_answer=entry.answer,
                ),
            )
            for i, entry in enumerate(entries)
        ]

        return Dataset[VQARadDataPoint](
            data_points=data_points, description=self.__doc__
        )
