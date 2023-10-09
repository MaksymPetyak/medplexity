from abc import ABC, abstractmethod

from medplexity.datasets.dataset import Dataset


class DatasetFactory(ABC):
    """Abstract class for creating datasets."""

    @abstractmethod
    def build_dataset(self, *args, **kwargs) -> Dataset:
        """Build and return the dataset."""
        pass
