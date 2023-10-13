from medplexity.benchmarks.pubmedqa import PubmedQADatasetBuilder
from medplexity.benchmarks.pubmedqa.pubmedqa_dataset_builder import PubMedQADatasetTypes
from tests.mock_loader import MockLoader

# Sample mocked data resembling the output from the loader.
SAMPLE_LOADER_OUTPUT = [
    {
        "question": "Are ILC2s increased in CRS?",
        "context": {
            "contexts": ["Context sentence 1.", "Context sentence 2."],
            "labels": ["BACKGROUND", "RESULTS"],
            "meshes": ["Adult", "Aged"],
        },
        "long_answer": "ILC2s are elevated in patients...",
        "final_decision": "yes",
    }
]


def test_build_dataset():
    # Instantiate the builder with the MockLoader
    builder = PubmedQADatasetBuilder(loader=MockLoader(SAMPLE_LOADER_OUTPUT))

    dataset = builder.build_dataset(config={"subset": PubMedQADatasetTypes.pqa_labeled})

    # Check the basic transformation is working
    assert len(dataset.data_points) == 1
    assert dataset.data_points[0].input.question == "Are ILC2s increased in CRS?"
    assert dataset.data_points[0].expected_output == "(A)"
    assert (
        dataset.data_points[0].metadata.explanation
        == "ILC2s are elevated in patients..."
    )
    assert dataset.data_points[0].metadata.labels == ["BACKGROUND", "RESULTS"]
    assert dataset.data_points[0].metadata.meshes == ["Adult", "Aged"]
