from typing import Optional, Callable

from tqdm import tqdm

from medplexity.chains.chain import Chain
from medplexity.datasets.dataset import Dataset
from medplexity.evaluators.evaluation_summary import EvaluationSummary, FailedEvaluation
from medplexity.evaluators.evaluator import Evaluator, EvaluationResult


class SequentialEvaluator(Evaluator):
    """Simply goes over the dataset one item and a time and calls the LLM on the input."""

    def __init__(
        self,
        chain: Chain,
        format_checker: Optional[Callable[[str], bool]] = None,
        comparator=None,
    ):
        self.chain = chain
        self.format_checker = format_checker
        self.comparator = comparator

    def evaluate(
        self,
        dataset: Dataset,
        max_items: Optional[int] = None,
        ignore_errors: bool = False,
    ) -> EvaluationSummary:
        results = EvaluationSummary(
            evaluation_results=[],
            failed_evaluations=[],
        )

        count = 0
        for data_point in tqdm(dataset):
            if max_items and count > max_items:
                break

            try:
                llm_output = self.chain(data_point.input)

                if self.format_checker:
                    is_format_correct = self.format_checker(llm_output)
                    if not is_format_correct:
                        results.undefined.append(data_point)
                        continue

                are_outputs_equal = False
                if self.comparator:
                    are_outputs_equal = self.comparator(
                        data_point.expected_output, llm_output
                    )
                else:
                    are_outputs_equal = data_point.expected_output == llm_output

                results.evaluation_results.append(
                    EvaluationResult(
                        input=data_point.input,
                        input_metadata=data_point.metadata,
                        expected_output=data_point.expected_output,
                        output=llm_output,
                        correct=are_outputs_equal,
                    )
                )
            except Exception as e:
                if ignore_errors:
                    results.failed_evaluations.append(
                        FailedEvaluation(
                            datapoint=data_point,
                            error=str(e),
                        )
                    )
                else:
                    raise e

            count += 1

        return results
