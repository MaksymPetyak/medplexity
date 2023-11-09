from medplexity.prompts.prompt import Prompt


class ConversationSummarisationPromptTemplate(Prompt):
    """Example prompt for long-form question answering, from https://arxiv.org/abs/2305.09617"""

    PROMPT = """You are a helpful medical knowledge assistant. Please summarise a given conversation into a short clinical note.
Input: {conversation}
Summary: """

    def format(self, conversation: str):
        return self.PROMPT.format(
            conversation=conversation,
        )
