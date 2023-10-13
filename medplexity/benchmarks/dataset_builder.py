from abc import ABC
from typing import Dict, Any

from medplexity.benchmarks.loaders import Loader, HuggingFaceLoader
from medplexity.datasets.dataset import Dataset


class DatasetBuilder(ABC):
    """Abstract class for creating datasets."""

    def __init__(self, loader: Loader = None):
        if loader is None:
            loader = HuggingFaceLoader()

        self.loader = loader

    def build_dataset(
        self, split_type: str, config: Dict[str, Any] | None = None
    ) -> Dataset:
        """Build and return the dataset."""
        pass
