from medplexity.benchmarks.medmcqa import MedMCQADatasetBuilder
from tests.mock_loader import MockLoader

SAMPLE_MEDMCQA_LOADER_OUTPUT = [
    {
        "question": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma",
        "opa": "Hyperplasia",
        "opb": "Hyperophy",
        "opc": "Atrophy",
        "opd": "Dyplasia",
        "cop": "2c",
        "exp": "Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950",
        "subject_name": "Anatomy",
    }
]


def test_medmcqa_build_dataset():
    # Instantiate the builder with the MockLoader
    builder = MedMCQADatasetBuilder(loader=MockLoader(SAMPLE_MEDMCQA_LOADER_OUTPUT))

    dataset = builder.build_dataset(split_type="train")

    # Check the transformation for the question
    assert len(dataset.data_points) == 1
    assert (
        dataset.data_points[0].input.question
        == "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma"
    )
    assert dataset.data_points[0].input.options == [
        "Hyperplasia",
        "Hyperophy",
        "Atrophy",
        "Dyplasia",
    ]
    assert dataset.data_points[0].expected_output == "(C)"
    assert (
        dataset.data_points[0].metadata.explanation
        == "Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950"
    )
    assert dataset.data_points[0].metadata.subject_name == "Anatomy"
