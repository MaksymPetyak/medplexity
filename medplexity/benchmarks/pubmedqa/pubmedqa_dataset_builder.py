from enum import Enum
from pathlib import Path
from typing import Literal, List

from pydantic import BaseModel

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.multiple_choice_utils import (
    MultipleChoiceInput,
    format_answer_to_letter,
)
from medplexity.benchmarks.pubmedqa.models import PubMedQAQuestion
from medplexity.datasets.dataset import DataPoint, Dataset

# Only train split is available
PubMedQADatasetSplitType = Literal["train"]


# config types
# "pqa_artificial"
# "pqa_labeled"
# "pqa_unlabeled"
class PubMedQADatasetTypes(str, Enum):
    pqa_artificial = "pqa_artificial"
    pqa_labeled = "pqa_labeled"
    pqa_unlabeled = "pqa_unlabeled"


class PubmedQAMetadata(BaseModel):
    explanation: str | None
    labels: List[str]
    meshes: List[str]


class PubmedQADataPoint(DataPoint):
    input: MultipleChoiceInput
    expected_output: str
    metadata: PubmedQAMetadata


class PubmedQADatasetBuilder(DatasetBuilder):
    """PubMedQA is a biomedical QA dataset designed to answer research questions with yes/no/maybe. The dataset consists of 1k expert-annotated questions, 61.2k unlabeled questions, and an additional 211.3k artificially generated QA instances. Every instance contains a question sourced or derived from a research article title, context from the abstract without its conclusion, a long answer in the form of the abstract's conclusion, and a summarized yes/no/maybe answer.

    Paper: PubMedQA: A Dataset for Biomedical Research Question Answering

    13 Sep 2019 · Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, Xinghua Lu
    <https://arxiv.org/abs/1909.06146>

    Only train split available.

    Divided into three subsets:
    - pqa_artificial: 211.3k artificially generated QA instances
    - pqa_labeled: 1k expert-annotated questions
    - pqa_unlabeled: 61.2k unlabeled questions

    Dataset version used: <https://huggingface.co/datasets/pubmed_qa>
    """

    EXAMPLE_QUESTIONS_PATH = Path(__file__).resolve().parent / "examples.json"

    def build_dataset(
        self, split_type: PubMedQADatasetSplitType = "train", config=None
    ) -> Dataset[PubmedQADataPoint]:
        if config is None:
            config = {"subset": PubMedQADatasetTypes.pqa_labeled}

        dataset = self.loader.load("pubmed_qa", config["subset"], split=split_type)

        questions = [PubMedQAQuestion(**row) for row in dataset]

        options = ["Yes", "No", "Maybe"]

        data_points = [
            PubmedQADataPoint(
                id=f"{split_type}-{i}",
                input=MultipleChoiceInput(
                    question=question.question,
                    options=options,
                    context=" ".join(question.context.contexts),
                ),
                expected_output=format_answer_to_letter(
                    options, question.final_decision.value.capitalize().strip()
                ),
                metadata=PubmedQAMetadata(
                    explanation=question.long_answer,
                    labels=question.context.labels,
                    meshes=question.context.meshes,
                ),
            )
            for i, question in enumerate(questions)
        ]

        return Dataset[PubmedQADataPoint](
            data_points=data_points, description=self.__doc__
        )
