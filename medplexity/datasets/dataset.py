from typing import Any, List, Generic, TypeVar

from pydantic import BaseModel

DataT = TypeVar("DataT")


class DataPoint(BaseModel, Generic[DataT]):
    id: str
    input: DataT
    expected_output: Any
    metadata: Any

    def __repr__(self) -> str:
        return self.model_dump_json(indent=4)


DataPointT = TypeVar("DataPointT")


class Dataset(Generic[DataPointT]):
    """Dataset objects can be iterated through and evaluated against"""

    def __init__(self, data_points: List[DataPointT], description: str = ""):
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

    def __getitem__(self, idx) -> DataPointT:
        return self.data_points[idx]
