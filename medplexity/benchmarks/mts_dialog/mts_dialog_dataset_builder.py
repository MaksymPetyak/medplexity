from typing import Literal

from pydantic import BaseModel

from medplexity.benchmarks.dataset_builder import DatasetBuilder
from medplexity.benchmarks.loaders import Loader
from medplexity.benchmarks.utils import parse_csv_from_url
from medplexity.datasets.dataset import DataPoint, Dataset
from medplexity.benchmarks.mts_dialog.models import MTSDialogEntry


MTSDialogDatasetSplitType = Literal["train", "validation", "test-1", "test-2"]


class MTSDialogInput(BaseModel):
    dialog: str


class MTSDialogMetadata(BaseModel):
    section_header: str
    reference_summary: str


class MTSDialogDataPoint(DataPoint):
    input: MTSDialogInput
    # No expected output as summaries will be different we avoid comparison like this
    expected_output: None


class MTSDialogGithubDatasetLoader(Loader):
    """Loader to downlaod the dataset from GitHub - https://github.com/abachaa/MTS-Dialog"""

    SPLIT_TYPE_TO_DATASET_URL = {
        "train": "https://github.com/abachaa/MTS-Dialog/raw/main/Main-Dataset/MTS-Dialog-TrainingSet.csv",
        "validation": "https://github.com/abachaa/MTS-Dialog/raw/main/Main-Dataset/MTS-Dialog-ValidationSet.csv",
        "test-1": "https://github.com/abachaa/MTS-Dialog/raw/main/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
        "test-2": "https://github.com/abachaa/MTS-Dialog/raw/main/Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv",
    }

    def load(self, split_type: MTSDialogDatasetSplitType = "test"):
        return parse_csv_from_url(self.SPLIT_TYPE_TO_DATASET_URL[split_type]).to_dict(
            orient="records"
        )


class MTSDialogDatasetBuilder(DatasetBuilder):
    """
    MTS-Dialog (Medical Training Summarization Dialog) is a comprehensive dataset featuring 1.7k doctor-patient conversations, along with their corresponding summaries, including section headers and contents.

    The dataset is structured as follows:

    - Training set: Comprises 1,201 pairs of conversations and associated summaries, aimed at facilitating the training of models for medical dialogue understanding and summarization.

    - Validation set: Contains 100 pairs of conversations and their summaries, used for model tuning and intermediate evaluation.

    - Test sets: Includes two distinct test sets, each with 200 conversations and corresponding section headers and contents:

        1. MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv: Serves as the official test set for the MEDIQA-Chat 2023 challenge (Task A), focusing on chat-based medical consultations.

        2. MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv: Used as the official test set for the MEDIQA-Sum 2023 challenge (Task A & Task B), emphasizing the summary generation from medical dialogues.

    Paper: "MTS-Dialog: A New Dataset for Medical Training Summarization in Doctor-Patient Conversations" - <https://aclanthology.org/2023.eacl-main.1681>

    Authors: Asma Ben Abacha, Wen-wai Yim, Yadan Fan, Thomas Lin

    Dataset version from the GitHub repository: <https://github.com/abachaa/MTS-Dialog>
    """

    def __init__(self, loader: Loader = None):
        if loader is None:
            loader = MTSDialogGithubDatasetLoader()

        super().__init__(loader)

    def build_dataset(
        self,
        split_type: MTSDialogDatasetSplitType = "test",
        config=None,
    ) -> Dataset[MTSDialogDataPoint]:
        dialog_raw_data = self.loader.load(split_type)

        dialog_entries = [
            MTSDialogEntry(**dialog_raw) for dialog_raw in dialog_raw_data
        ]

        data_points = [
            MTSDialogDataPoint(
                id=str(dialog_entry.ID),
                input=MTSDialogInput(
                    dialog=dialog_entry.dialogue,
                ),
                expected_output=None,
                metadata=MTSDialogMetadata(
                    section_header=dialog_entry.section_header,
                    reference_summary=dialog_entry.section_text,
                ),
            )
            for dialog_entry in dialog_entries
        ]

        return Dataset[MTSDialogDataPoint](
            data_points=data_points, description=self.__doc__
        )
