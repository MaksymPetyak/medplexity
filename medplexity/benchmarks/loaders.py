import abc

from datasets import load_dataset


class Loader(abc.ABC):
    """Abstract class for loading dataset in the original format."""

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        raise NotImplementedError


class HuggingFaceLoader(Loader):
    """Loader for datasets from HuggingFace's datasets library."""

    def load(self, *args, **kwargs):
        return load_dataset(*args, **kwargs)
