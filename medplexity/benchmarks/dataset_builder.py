from abc import ABC, abstractmethod

from medplexity.datasets.dataset import Dataset


class DatasetBuilder(ABC):
    """Abstract class for dataset builders."""

    @abstractmethod
    def build_dataset(self, *args, **kwargs) -> Dataset:
        """Build and return the dataset."""
        pass
