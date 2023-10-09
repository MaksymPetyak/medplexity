from typing import Literal, List

from pydantic import BaseModel

from medplexity.benchmarks.dataset_factory import DatasetFactory
from medplexity.benchmarks.pubmedqa.models import PubmedQADecision, PubMedQAQuestion
from medplexity.datasets.dataset import DataPoint, Dataset

from datasets import load_dataset

# Only train split is available
PubMedQADatasetSplitType = Literal["train"]

# config types
# "pqa_artificial"
# "pqa_labeled"
# "pqa_unlabeled"
PubMedQADatasetConfigType = Literal["pqa_artificial", "pqa_labeled", "pqa_unlabeled"]


class PubmedQAInput(BaseModel):
    question: str
    contexts: list[str]


class PubmedQAMetadata(BaseModel):
    explanation: str | None
    labels: List[str]
    meshes: List[str]


class PubmedQADataPoint(DataPoint):
    input: PubmedQAInput
    expected_output: PubmedQADecision
    metadata: PubmedQAMetadata


class PubmedQADatasetFactory(DatasetFactory):
    """PubMedQA is a biomedical QA dataset designed to answer research questions with yes/no/maybe. The dataset consists of 1k expert-annotated questions, 61.2k unlabeled questions, and an additional 211.3k artificially generated QA instances. Every instance contains a question sourced or derived from a research article title, context from the abstract without its conclusion, a long answer in the form of the abstract's conclusion, and a summarized yes/no/maybe answer.

    Original paper: PubMedQA: A Dataset for Biomedical Research Question Answering

    13 Sep 2019 · Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, Xinghua Lu
    <https://arxiv.org/abs/1909.06146>

    Only train split available.

    Divided into three subsets:
    - pqa_artificial: 211.3k artificially generated QA instances
    - pqa_labeled: 1k expert-annotated questions
    - pqa_unlabeled: 61.2k unlabeled questions

    Dataset version used: `https://huggingface.co/datasets/pubmed_qa`
    """

    def build_dataset(
        self,
        config_type: PubMedQADatasetConfigType = "pqa_labeled",
        split_type: PubMedQADatasetSplitType = "train",
    ) -> Dataset[PubmedQADataPoint]:
        dataset = load_dataset("pubmed_qa", config_type, split=split_type)

        questions = [PubMedQAQuestion(**row) for row in dataset]

        data_points = [
            PubmedQADataPoint(
                input=PubmedQAInput(
                    question=question.question,
                    contexts=question.context.contexts,
                ),
                expected_output=question.final_decision,
                metadata=PubmedQAMetadata(
                    explanation=question.long_answer,
                    labels=question.context.labels,
                    meshes=question.context.meshes,
                ),
            )
            for question in questions
        ]

        return Dataset[PubmedQADataPoint](
            data_points=data_points, description=self.__doc__
        )
