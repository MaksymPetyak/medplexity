from typing import Any

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    input: Any
    input_metadata: Any
    expected_output: Any

    output: Any

    correct: bool

class FailedEvaluation(BaseModel):
    datapoint: Any
    error: str

class EvaluationSummary(BaseModel):
    evaluation_results: list[EvaluationResult]
    failed_evaluations: list[FailedEvaluation]

    def accuracy(self):
        correct, incorrect = self.partition_by_correctness()
        return len(correct) / (len(correct) + len(incorrect))

    def partition_by_correctness(self):
        correct = []
        incorrect = []
        for result in self.evaluation_results:
            if result.correct:
                correct.append(result)
            else:
                incorrect.append(result)

        return correct, incorrect
