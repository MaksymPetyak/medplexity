from medplexity.chains.chain import Chain
from medplexity.datasets.dataset import Dataset
from medplexity.evaluators.evaluation_summary import EvaluationSummary
from medplexity.evaluators.evaluator import Evaluator
from medplexity.evaluators.sequential_evaluator import SequentialEvaluator


class Medharness:
    """Medharness main class for evaluation of given dataset and chain."""

    def __init__(
        self,
        dataset: Dataset,
        chain: Chain,
    ):
        self.dataset = dataset
        self.chain = chain

        self._result: EvaluationSummary | None = None

    def run(
        self,
        k: int | None = None,
        evaluator: Evaluator | None = None,
        ignore_errors: bool = False,
    ) -> EvaluationSummary:
        """Run the evaluation on the dataset with the provided chain. Results are stored in self.result.

        Args:
            k (int | None, optional): Limit the dataset to the first k items.
                Defaults to None, which means the entire dataset will be used.
            evaluator (Evaluator | None, optional): Evaluator to be used for the evaluation.
                Defaults to SequentialEvaluator.

        Returns:
            EvaluationSummary: Results of the evaluation.
        """
        if evaluator is None:
            evaluator = SequentialEvaluator()

        if k:
            dataset = self.dataset[:k]
        else:
            dataset = self.dataset

        result = evaluator.evaluate(
            dataset=dataset, chain=self.chain, ignore_errors=ignore_errors
        )

        self.result = result

        return result

    def save_results(self, filename: str, additional_data: dict = None):
        if self.result is None:
            raise ValueError("No result to save.")

        self.result.save(filename, additional_data=additional_data)

    @property
    def result(self) -> EvaluationSummary:
        """Report the results of the benchmarks."""
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
