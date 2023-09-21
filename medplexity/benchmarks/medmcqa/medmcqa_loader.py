import json
from pathlib import Path
from typing import Literal, List

from datasets import load_dataset

from medplexity.benchmarks.medmcqa.models import MedMCQAQuestion


DATASET_TYPE = Literal["train", "test", "validation"]


class MedMCQALoader():
    """Class to load questions from https://github.com/medmcqa/medmcqa"""

    def __init__(self, dataset_directory: str | Path | None = None):
        # TODO: download the dataset if it's not there
        if dataset_directory is not None:
            if not isinstance(dataset_directory, Path):
                dataset_directory = Path(dataset_directory)
            if not dataset_directory.exists():
                raise ValueError(f"{str(dataset_directory)} doesn't exist")

        self.dataset_directory: Path = dataset_directory

    def load_questions(self, dataset_type: DATASET_TYPE) -> List[MedMCQAQuestion]:
        if self.dataset_directory is None:
            return self.download_dataset_and_load_questions(dataset_type)
        else:
            return self.load_questions_from_file(dataset_type)


    def download_dataset_and_load_questions(self, dataset_type: DATASET_TYPE):
        medmcqa_dataset = load_dataset("medmcqa", split=dataset_type)

        questions = [MedMCQAQuestion(**row) for row in medmcqa_dataset]

        return questions

    def load_questions_from_file(self, dataset_type: DATASET_TYPE) -> List[MedMCQAQuestion]:
        file_path = self.dataset_directory / Path(f"{dataset_type}.json")
        if not file_path.exists():
            raise ValueError(f"Path to file {str(file_path)} doesn't exist")

        with open(file_path, 'r') as f:
            raw_data = f.read()

            # the file is JSON but delimited by new-line
            lines = [line for line in raw_data.split('\n') if line]
            objects = [json.loads(line) for line in lines]

        questions = [MedMCQAQuestion(**object) for object in objects]

        return questions

