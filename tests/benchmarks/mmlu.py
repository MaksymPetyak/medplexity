# Sample mocked data resembling the output from the loader for MMLU.
from medplexity.benchmarks.mmlu import MMLUDatasetBuilder
from medplexity.benchmarks.mmlu.mmlu_dataset_builder import MMLUSubsetConfig
from tests.mock_loader import MockLoader

SAMPLE_MMLU_LOADER_OUTPUT = [
    {
        "input": "Where do most short-period comets come from and how do we know?",
        "A": "The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.",
        "B": "The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.",
        "C": "The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.",
        "D": "The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.",
        "target": "A",
    }
]


def test_mmlu_build_dataset():
    # Instantiate the builder with the MockLoader
    builder = MMLUDatasetBuilder(loader=MockLoader(SAMPLE_MMLU_LOADER_OUTPUT))

    dataset = builder.build_dataset(
        split_type="train", config={"subset": MMLUSubsetConfig.clinical_knowledge}
    )

    # Check the transformation for the question
    assert len(dataset.data_points) == 1
    assert (
        dataset.data_points[0].input.question
        == "Where do most short-period comets come from and how do we know?"
    )
    assert dataset.data_points[0].input.options == [
        "The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.",
        "The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.",
        "The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.",
        "The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.",
    ]
    assert dataset.data_points[0].expected_output == "(A)"
