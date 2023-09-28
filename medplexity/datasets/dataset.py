from typing import Any, List

from pydantic import BaseModel


class DataPoint(BaseModel):
    input: Any
    expected_output: Any
    metadata: Any


class Dataset:
    """Dataset objects can be iterated through and evaluated against"""

    def __init__(self, data_points: List[DataPoint], description: str = ""):
        """
        Initialize the dataset with data points.

        Parameters:
        - data_points: List of data points
        """

        self.data_points = data_points
        self._description = description

    def __iter__(self):
        """
        Return the iterator object.
        """
        return iter(self.data_points)

    def __len__(self):
        return len(self.data_points)

    @property
    def description(self):
        return self._description

    def __getitem__(self, idx):
        return self.data_points[idx]
