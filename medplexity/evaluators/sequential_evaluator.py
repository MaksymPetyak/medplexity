from typing import Optional, Callable

from tqdm import tqdm

from medplexity.chains.chain import Chain, ChainOutput
from medplexity.datasets.dataset import Dataset
from medplexity.evaluators.evaluation_summary import EvaluationSummary, FailedEvaluation
from medplexity.evaluators.evaluator import Evaluator, EvaluationResult


class SequentialEvaluator(Evaluator):
    """Simply goes over the dataset one item at a time and calls the LLM on the input."""

    def __init__(
        self,
        format_checker: Optional[Callable[[str], bool]] = None,
        comparator=None,
    ):
        self.format_checker = format_checker
        self.comparator = comparator

    def evaluate(
        self,
        dataset: Dataset,
        chain: Chain,
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
                chain_output = chain(data_point.input)
                output_metadata = None

                if isinstance(chain_output, ChainOutput):
                    output_metadata = chain_output.output_metadata
                    chain_output = chain_output.output

                if self.format_checker:
                    is_format_correct = self.format_checker(chain_output)
                    if not is_format_correct:
                        results.failed_evaluations.append(data_point)
                        continue

                correct = None
                if data_point.expected_output is not None:
                    if self.comparator:
                        correct = self.comparator(
                            data_point.expected_output, chain_output
                        )
                    else:
                        correct = data_point.expected_output == chain_output

                results.evaluation_results.append(
                    EvaluationResult(
                        id=data_point.id,
                        input=data_point.input,
                        input_metadata=data_point.metadata,
                        expected_output=data_point.expected_output,
                        output=chain_output,
                        output_metadata=output_metadata,
                        correct=correct,
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
