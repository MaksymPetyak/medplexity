from medplexity.prompts.prompt import Prompt


class MedicalAssistantPromptTemplate(Prompt):
    """Example prompt for long-form question answering, from https://arxiv.org/abs/2305.09617"""

    PROMPT = """You are a helpful medical knowledge assistant. Provide useful, complete, and scientifically-grounded answers to common
consumer search queries about health.

Question: {question}
Complete Answer:"""

    def format(self, question: str):
        return self.PROMPT.format(
            question=question,
        )
