import os
from typing import Dict, Any, Optional

import openai

from medplexity.llms.llm import LLM


class OpenAI(LLM):
    """OpenAI LLM using BaseOpenAI Class.

    An API call to OpenAI API is sent and response is recorded and returned.
    The default chat model is **gpt-3.5-turbo** while **text-davinci-003** is only
    supported completion model.
    The list of supported Chat models includes ["gpt-4", "gpt-4-0314", "gpt-4-32k",
     "gpt-4-32k-0314","gpt-3.5-turbo", "gpt-3.5-turbo-0301"].

    """

    api_token: str
    temperature: float = 0
    max_tokens: int = 1000
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0.6
    stop: Optional[str] = None
    # support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None

    _supported_chat_models = [
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
    ]
    _supported_completion_models = ["text-davinci-003"]

    model: str = "gpt-3.5-turbo"

    def __init__(
        self,
        api_token: Optional[str] = None,
        api_key_path: Optional[str] = None,
        **kwargs,
    ):
        """
        __init__ method of OpenAI Class

        Args:
            api_token (str): API Token for OpenAI platform.
            **kwargs: Extended Parameters inferred from BaseOpenAI class

        """
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None
        self.api_key_path = api_key_path

        if (not self.api_token) and (not self.api_key_path):
            raise ValueError("Either OpenAI API key or key path is required")

        if self.api_token:
            openai.api_key = self.api_token
        else:
            openai.api_key_path = self.api_key_path

        self.openai_proxy = kwargs.get("openai_proxy") or os.getenv("OPENAI_PROXY")
        if self.openai_proxy:
            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}

        self._set_params(**kwargs)

    def _set_params(self, **kwargs):
        """
        Set Parameters
        Args:
            **kwargs: ["model", "engine", "deployment_id", "temperature","max_tokens",
            "top_p", "frequency_penalty", "presence_penalty", "stop", ]

        Returns:
            None.

        """

        valid_params = [
            "model",
            "engine",
            "deployment_id",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def completion(self, prompt: str) -> str:
        params = {**self._default_params, "prompt": prompt}

        response = openai.Completion.create(**params)

        return response["choices"][0]["text"]

    def chat_completion(self, value: str) -> str:
        params = {
            **self._default_params,
            "messages": [
                {
                    "role": "system",
                    "content": value,
                }
            ],
        }

        response = openai.ChatCompletion.create(**params)

        return response["choices"][0]["message"]["content"]

    def __call__(self, instruction: str, suffix: str = "") -> str:
        """
        Call the OpenAI LLM.

        Args:
            instruction (Prompt): A prompt object with instruction for LLM.
            suffix (str): Suffix to pass.

        Returns:
            str: Response
        """
        self.last_prompt = instruction + suffix

        if self.model in self._supported_completion_models:
            response = self.completion(self.last_prompt)
        elif self.model in self._supported_chat_models:
            response = self.chat_completion(self.last_prompt)
        else:
            raise ValueError("Unsupported model")

        return response

    @property
    def type(self) -> str:
        return "openai"
