from medplexity.benchmarks.healthsearchqa import HealthSearchQADatasetBuilder
from tests.mock_loader import MockLoader

SAMPLE_HEALTHSEARCHQA_LOADER_OUTPUT = [
    {
        "id": 1,
        "question": "What is losing balance a symptom of?",
    }
]


def test_healthsearchqa_build_dataset():
    # Instantiate the builder with the MockLoader
    builder = HealthSearchQADatasetBuilder(
        loader=MockLoader(SAMPLE_HEALTHSEARCHQA_LOADER_OUTPUT)
    )

    dataset = builder.build_dataset(split_type="train")

    # Check the transformation for the question
    assert len(dataset.data_points) == 1
    assert dataset.data_points[0].input == "What is losing balance a symptom of?"
    # Since there's no expected output or metadata, we can assert those are None
    assert dataset.data_points[0].expected_output is None
    assert dataset.data_points[0].metadata is None
