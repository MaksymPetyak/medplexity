from pydantic import BaseModel

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.medicationqa.models import MedicationQAEntry
from medplexity.datasets.dataset import Dataset, DataPoint


class MedicationQAMetaData(BaseModel):
    # Example answer
    answer: str | None
    focus: str | None
    question_type: str | None
    section_title: str | None
    url: str | None


class MedicationQADataPoint(DataPoint):
    input: str
    expected_output: None
    metadata: MedicationQAMetaData


class MedicationQADatasetBuilder(DatasetBuilder):
    """Medication Question Answering created using real consumer questions.

    Paper: Bridging the Gap between Consumers’ Medication Questions and Trusted Answers.
    2019 * Asma Ben Abacha, Yassine Mrabet, Mark Sharp, Travis Goodwin, Sonya E. Shooshan and Dina Demner-Fushman
    <http://ebooks.iospress.nl/publication/51941>

    No dataset splitting (only "train" split).

    Dataset version used: <https://huggingface.co/datasets/truehealth/medicationqa/viewer/default/train>
    """

    def build_dataset(
        self,
        split_type: str = "train",
        config=None,
    ) -> Dataset[MedicationQADataPoint]:
        dataset = self.loader.load("truehealth/medicationqa", split=split_type)

        questions = [MedicationQAEntry(**row) for row in dataset]

        data_points = [
            MedicationQADataPoint(
                id=f"{split_type}-{i}",
                input=question.question,
                expected_output=None,
                metadata=MedicationQAMetaData(
                    answer=question.answer,
                    focus=question.focus,
                    question_type=question.question_type,
                    section_title=question.section_title,
                    url=question.url,
                ),
            )
            for i, question in enumerate(questions)
        ]

        return Dataset[MedicationQADataPoint](
            data_points=data_points, description=self.__doc__
        )
