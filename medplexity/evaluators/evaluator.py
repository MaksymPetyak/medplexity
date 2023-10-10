import abc
from typing import Optional

from medplexity.chains.chain import Chain
from medplexity.datasets.dataset import Dataset
from medplexity.evaluators.evaluation_summary import EvaluationResult


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self,
        dataset: Dataset,
        chain: Chain,
        max_items: Optional[int] = None,
        ignore_errors: bool = False,
    ) -> EvaluationResult:
        """Evaluate the LLM on the dataset and return the results."""
        raise NotImplementedError("Evaluator evaluate method is not implemented")
