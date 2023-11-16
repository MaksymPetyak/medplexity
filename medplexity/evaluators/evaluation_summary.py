import json
from typing import Any

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    id: str
    input: Any
    input_metadata: Any
    expected_output: Any

    output: Any
    output_metadata: Any

    # Can be None if the output is not comparable
    correct: bool | None

    def __repr__(self):
        return self.model_dump_json(indent=4)


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

    def save(self, filename: str, additional_data: dict = None):
        with open(filename, "w") as file:
            json_obj = json.loads(self.model_dump_json())

            if additional_data is not None:
                json_obj.update(additional_data)

            file.write(json.dumps(json_obj))

    @classmethod
    def load_from_file(cls, filename: str):
        with open(filename, "r") as file:
            data = json.load(file)
            return cls(**data)
