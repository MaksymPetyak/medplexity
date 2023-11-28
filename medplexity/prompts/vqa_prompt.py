from medplexity.prompts.prompt import Prompt


class VQAPromptTemplate(Prompt):
    PROMPT = """You are a helpful medical knowledge assistant. Please answer the provided question about the given image. Don't give any disclaimers or caveats, just answer the question.
Question: {question}"""

    def format(self, question: str):
        return self.PROMPT.format(
            question=question,
        )
