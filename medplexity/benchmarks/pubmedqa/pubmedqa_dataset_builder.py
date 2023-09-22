from typing import Literal, List

from pydantic import BaseModel

from medplexity.benchmarks.pubmedqa.models import PubmedQADecision, PubMedQAQuestion
from medplexity.benchmarks.pubmedqa.pubmedqa_prompt_template import (
    PUBMEDQA_ANSWER_TO_OPTION,
)
from medplexity.datasets.dataset import DataPoint

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
    expected_output: Literal["(A)", "(B)", "(C)"]
    metadata: PubmedQAMetadata


class PubmedQADatasetBuilder:
    def __init__(self):
        pass

    def build_dataset(
        self,
        config_type: PubMedQADatasetConfigType = "pqa_labeled",
        split_type: PubMedQADatasetSplitType = "train",
    ) -> list[PubmedQADataPoint]:
        dataset = load_dataset("pubmed_qa", config_type, split=split_type)

        questions = [PubMedQAQuestion(**row) for row in dataset]

        data_points = [
            PubmedQADataPoint(
                input=PubmedQAInput(
                    question=question.question,
                    contexts=question.context.contexts,
                ),
                expected_output=PUBMEDQA_ANSWER_TO_OPTION[
                    question.final_decision.lower().strip()
                ],
                metadata=PubmedQAMetadata(
                    explanation=question.long_answer,
                    labels=question.context.labels,
                    meshes=question.context.meshes,
                ),
            )
            for question in questions
        ]

        return data_points
