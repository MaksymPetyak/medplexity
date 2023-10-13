from medplexity.benchmarks.medqa import MedQADatasetBuilder
from medplexity.benchmarks.medqa.medqa_dataset_builder import MedQASubsetConfig
from tests.mock_loader import MockLoader

SAMPLE_MEDQA_LOADER_OUTPUT = [
    {
        "id": "0",
        "question_id": "0",
        "document_id": "0",
        "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. Which of the following is the best treatment for this patient?",
        "type": "multiple_choice",
        "choices": [
            "Ampicillin",
            "Ceftriaxone",
            "Ciprofloxacin",
            "Doxycycline",
            "Nitrofurantoin",
        ],
        "context": "",
        "answer": ["Nitrofurantoin"],
    }
]


def test_medqa_build_dataset():
    # Instantiate the builder with the MockLoader
    builder = MedQADatasetBuilder(loader=MockLoader(SAMPLE_MEDQA_LOADER_OUTPUT))

    dataset = builder.build_dataset(
        split_type="train", config={"subset": MedQASubsetConfig.med_qa_en_bigbio_qa}
    )

    # Check the transformation for the question
    assert len(dataset.data_points) == 1
    assert (
        dataset.data_points[0].input.question
        == "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. Which of the following is the best treatment for this patient?"
    )
    assert dataset.data_points[0].input.options == [
        "Ampicillin",
        "Ceftriaxone",
        "Ciprofloxacin",
        "Doxycycline",
        "Nitrofurantoin",
    ]
    assert dataset.data_points[0].expected_output == "(E)"
