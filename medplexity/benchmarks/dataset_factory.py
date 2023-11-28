from medplexity.benchmarks.healthsearchqa.healthsearchqa_dataset_builder import (
    HealthSearchQADatasetBuilder,
)
from medplexity.benchmarks.medicationqa.medicationqa_dataset_builder import (
    MedicationQADatasetBuilder,
)
from medplexity.benchmarks.medmcqa.medmcqa_dataset_builder import MedMCQADatasetBuilder
from medplexity.benchmarks.medqa.medqa_dataset_builder import MedQADatasetBuilder
from medplexity.benchmarks.mmlu.mmlu_dataset_builder import MMLUDatasetBuilder
from medplexity.benchmarks.mts_dialog import MTSDialogDatasetBuilder
from medplexity.benchmarks.pubmedqa.pubmedqa_dataset_builder import (
    PubmedQADatasetBuilder,
)
from medplexity.benchmarks.vqarad import VQARadDatasetBuilder


class DatasetFactory:
    DATASET_NAME_TO_BUILDER = {
        "healthsearchqa": HealthSearchQADatasetBuilder(),
        "medicationqa": MedicationQADatasetBuilder(),
        "medmcqa": MedMCQADatasetBuilder(),
        "medqa": MedQADatasetBuilder(),
        "mmlu": MMLUDatasetBuilder(),
        "pubmedqa": PubmedQADatasetBuilder(),
        "mts-dialog": MTSDialogDatasetBuilder(),
        "vqarad": VQARadDatasetBuilder(),
    }

    def build(self, name: str, split_type: str, config: dict | None = None):
        if name not in self.DATASET_NAME_TO_BUILDER:
            raise ValueError(f"Dataset {name} not supported.")

        return self.DATASET_NAME_TO_BUILDER[name].build_dataset(split_type, config)
