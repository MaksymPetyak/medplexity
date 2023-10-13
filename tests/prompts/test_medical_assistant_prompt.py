from medplexity.prompts.medical_assistant_prompt_template import (
    MedicalAssistantPromptTemplate,
)


def test_format_method_basic_usage():
    prompt = MedicalAssistantPromptTemplate()
    question = "What are the symptoms of a cold?"

    assert "Question: What are the symptoms of a cold?" in prompt.format(question)
