from medplexity.benchmarks.mts_dialog.mts_dialog_dataset_builder import (
    MTSDialogDatasetBuilder,
)
from tests.mock_loader import MockLoader

# Sample data for the MTSDialog loader
SAMPLE_MTSDIALOG_LOADER_OUTPUT = [
    {
        "ID": 0,
        "dialogue": "Doctor: Hi, how are you? \nPatient: I burned my hand.\nDoctor: Oh, I am sorry. Wow!\nPatient: Yeah.\nDoctor: Is it only right arm?\nPatient: Yes.",
        "section_text": "Burn, right arm.",
        "section_header": "CC",
    }
]


def test_mtsdialog_build_dataset():
    builder = MTSDialogDatasetBuilder(loader=MockLoader(SAMPLE_MTSDIALOG_LOADER_OUTPUT))

    dataset = builder.build_dataset(split_type="train")

    expected_datapoint = SAMPLE_MTSDIALOG_LOADER_OUTPUT[0]

    assert len(dataset.data_points) == 1
    assert dataset.data_points[0].input.dialog == expected_datapoint["dialogue"]
    assert dataset.data_points[0].expected_output == expected_datapoint["section_text"]
    assert (
        dataset.data_points[0].metadata.section_header
        == expected_datapoint["section_header"]
    )
