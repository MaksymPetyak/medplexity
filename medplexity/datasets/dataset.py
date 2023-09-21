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
        Initialize the dataset with data and labels.

        Parameters:
        - data: List of data points
        - labels: List of labels corresponding to the data points
        """

        self.index = 0
        self.data_points = data_points

        self._description = description

    def __iter__(self):
        """
        Return the iterator object (in this case, self).
        """
        return self

    def __len__(self):
        return len(self.data_points)

    @property
    def description(self):
        return self._description

    def __next__(self) -> DataPoint:
        """
        Return the next data point and label.
        """
        if self.index >= len(self):
            raise StopIteration

        data_point = self.data_points[self.index]

        self.index += 1

        return data_point
