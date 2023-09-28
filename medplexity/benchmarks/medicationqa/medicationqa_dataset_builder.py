from datasets import load_dataset
from pydantic import BaseModel

from benchmarks.medicationqa.models import MedicationQAEntry
from medplexity.datasets.dataset import Dataset, DataPoint


class MedicationQAInput(BaseModel):
    question: str


class MedicationQAMetaData(BaseModel):
    # Example answer
    answer: str | None
    focus: str | None
    question_type: str | None
    section_title: str | None
    url: str | None


class MedicationQADataPoint(DataPoint):
    input: MedicationQAInput
    expected_output: None
    metadata: MedicationQAMetaData


class MedicationQADatasetBuilder:
    """Medication Question Answering created using real consumer questions.

    Paper: Bridging the Gap between Consumers’ Medication Questions and Trusted Answers.
    2019 * Asma Ben Abacha, Yassine Mrabet, Mark Sharp, Travis Goodwin, Sonya E. Shooshan and Dina Demner-Fushman
    <http://ebooks.iospress.nl/publication/51941>

    No dataset splitting (only "train" split).

    Dataset version used: <https://huggingface.co/datasets/truehealth/medicationqa/viewer/default/train>
    """

    def build_dataset(
        self,
    ) -> Dataset:
        # No splitting, so just set split='train'
        dataset = load_dataset("truehealth/medicationqa", split="train")

        print(dataset[0])
        questions = [MedicationQAEntry(**row) for row in dataset]

        data_points = [
            MedicationQADataPoint(
                input=MedicationQAInput(
                    question=question.question,
                ),
                expected_output=None,
                metadata=MedicationQAMetaData(
                    answer=question.answer,
                    focus=question.focus,
                    question_type=question.question_type,
                    section_title=question.section_title,
                    url=question.url,
                ),
            )
            for question in questions
        ]

        return Dataset(data_points=data_points, description=self.__doc__)
