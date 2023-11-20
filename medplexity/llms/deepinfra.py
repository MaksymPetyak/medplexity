import os
import requests
from typing import Dict, Any, Optional

from medplexity.llms.llm import LLM
from dotenv import load_dotenv

load_dotenv()


class Deepinfra(LLM):
    """LLM wrapper around models provided by deepinfra API:  https://deepinfra.com/"""

    MODELS_AND_ENDPOINTS = {
        "llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",
        "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    }

    endpoint_base = "https://api.deepinfra.com/v1/inference/"

    api_token: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    stream: bool = False

    def __init__(
        self,
        api_token: Optional[str] = None,
        model: str = "llama-2-70b-chat-hf",
        **kwargs,
    ):
        self.api_token = api_token or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_token:
            raise ValueError("Deepinfra API key is required")

        if model not in self.MODELS_AND_ENDPOINTS:
            raise ValueError(f"Model {model} is not supported")
        self.model = model

        self._set_params(**kwargs)

    def _set_params(self, **kwargs):
        valid_params = [
            "temperature",
            "max_new_tokens",
            "top_p",
            "top_k" "repetition_penalty",
            "stream",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stream": self.stream,
        }

    def _prepare_payload(self, input_text: str) -> Dict[str, Any]:
        return {
            "input": input_text,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "stream": self.stream,
        }

    def completion(self, prompt: str) -> str:
        payload = self._prepare_payload(prompt)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        endpoint = self.endpoint_base + self.MODELS_AND_ENDPOINTS[self.model]
        response = requests.post(endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise ValueError(
                f"API call failed with status {response.status_code}: {response.text}"
            )

        return response.json()["results"][0]["generated_text"]

    def __call__(self, instruction: str, suffix: str = "", image=None) -> str:
        self.last_prompt = instruction + suffix
        response = self.completion(self.last_prompt)
        return response

    @property
    def type(self) -> str:
        return "deepinfra"
