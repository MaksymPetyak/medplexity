from medplexity.benchmarks.medicationqa import MedicationQADatasetBuilder
from tests.mock_loader import MockLoader

SAMPLE_MEDICATIONQA_LOADER_OUTPUT = [
    {
        "Question": "how does rivastigmine and otc sleep medicine interact",
        "Focus (Drug)": "rivastigmine",
        "Question Type": "Interaction",
        "Answer": "tell your doctor and pharmacist what prescription and nonprescription medications, vitamins, nutritional supplements, and herbal products you are taking or plan to take. Be sure to mention any of the following: antihistamines; aspirin and other nonsteroidal anti-inflammatory medications (NSAIDs) such as ibuprofen (Advil, Motrin) and naproxen (Aleve, Naprosyn); bethanechol (Duvoid, Urecholine); ipratropium (Atrovent, in Combivent, DuoNeb); and medications for Alzheimer's disease, glaucoma, irritable bowel disease, motion sickness, ulcers, or urinary problems. Your doctor may need to change the doses of your medications or monitor you carefully for side effects.",
        "Section Title": "What special precautions should I follow?",
        "URL": "https://medlineplus.gov/druginfo/meds/a602009.html",
    }
]


def test_medicationqa_build_dataset():
    # Instantiate the builder with the MockLoader
    builder = MedicationQADatasetBuilder(
        loader=MockLoader(SAMPLE_MEDICATIONQA_LOADER_OUTPUT)
    )

    dataset = builder.build_dataset(split_type="train")

    # Check the transformation for the question
    assert len(dataset.data_points) == 1
    assert (
        dataset.data_points[0].input
        == "how does rivastigmine and otc sleep medicine interact"
    )
    assert (
        dataset.data_points[0].metadata.answer
        == "tell your doctor and pharmacist what prescription and nonprescription medications, vitamins, nutritional supplements, and herbal products you are taking or plan to take. Be sure to mention any of the following: antihistamines; aspirin and other nonsteroidal anti-inflammatory medications (NSAIDs) such as ibuprofen (Advil, Motrin) and naproxen (Aleve, Naprosyn); bethanechol (Duvoid, Urecholine); ipratropium (Atrovent, in Combivent, DuoNeb); and medications for Alzheimer's disease, glaucoma, irritable bowel disease, motion sickness, ulcers, or urinary problems. Your doctor may need to change the doses of your medications or monitor you carefully for side effects."
    )
    assert dataset.data_points[0].metadata.focus == "rivastigmine"
    assert dataset.data_points[0].metadata.question_type == "Interaction"
    assert (
        dataset.data_points[0].metadata.section_title
        == "What special precautions should I follow?"
    )
    assert (
        dataset.data_points[0].metadata.url
        == "https://medlineplus.gov/druginfo/meds/a602009.html"
    )
