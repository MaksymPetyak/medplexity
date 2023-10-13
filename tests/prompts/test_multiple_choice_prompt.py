import pytest

from medplexity.prompts.multiple_choice_prompt import MultipleChoiceChainOfThoughtPrompt


def test_format():
    prompt = MultipleChoiceChainOfThoughtPrompt()

    # Basic test
    formatted = prompt.format("What is the capital of France?", ["Paris", "Berlin"])
    assert "Question: What is the capital of France?" in formatted
    assert "(A) Paris" in formatted
    assert "(B) Berlin" in formatted

    # With context and no examples
    formatted = prompt.format(
        "What is the capital of France?",
        ["Paris", "Berlin"],
        context="France is a country in Europe.",
    )
    assert "France is a country in Europe." in formatted
    assert "Use the following format for your answer:" in formatted


def test_format_example():
    prompt = MultipleChoiceChainOfThoughtPrompt()

    example = prompt.format_example(
        "France is a country.",
        "What is the capital?",
        ["Paris", "Berlin"],
        "(A)",
        "Paris is the capital of France.",
    )
    assert "France is a country." in example
    assert "(A) Paris" in example
    assert "Paris is the capital of France." in example


def test_format_options():
    prompt = MultipleChoiceChainOfThoughtPrompt()

    options_str = prompt.format_options(["A", "B"])
    assert options_str == "(A) A (B) B"


def test_extract_explanation_and_answer():
    prompt = MultipleChoiceChainOfThoughtPrompt()

    result = prompt.extract_explanation_and_answer(
        "Explanation: Paris is the capital. Answer: (A)"
    )
    assert result.answer == "(A)"
    assert result.explanation == "Paris is the capital."

    # Incorrect format
    with pytest.raises(ValueError):
        prompt.extract_explanation_and_answer(
            "Explanation: Paris is the capital. Ans: Paris"
        )
