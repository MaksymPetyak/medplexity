from typing import Dict, Any, Optional, Literal
from PIL.Image import Image

from openai import Client

from medplexity.llms.llm import LLM
from dotenv import load_dotenv

from medplexity.llms.utils import encode_image_base_64

load_dotenv()

ImageDetailLevel = Literal["low", "high", "auto"]


class OpenAI(LLM):
    """OpenAI LLM using BaseOpenAI Class.

    An API call to OpenAI API is sent and response is recorded and returned.
    The default chat model is **gpt-3.5-turbo**.
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
    detail_level: ImageDetailLevel | None = None

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

    _supported_vision_models = [
        "gpt-4-vision-preview",
    ]

    model: str = "gpt-3.5-turbo"

    def __init__(
        self,
        api_token: Optional[str] = None,
        detail_level: ImageDetailLevel | None = None,
        **kwargs,
    ):
        """
        __init__ method of OpenAI Class

        Args:
            api_token (str): API Token for OpenAI platform.
            **kwargs: Extended Parameters inferred from BaseOpenAI class

        """
        self._set_params(**kwargs)
        self.detail_level = detail_level

        self._client = Client(
            api_key=api_token,
        )

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

    def chat_completion(self, value: str) -> str:
        completion = self._client.chat.completions.create(
            **self._default_params,
            messages=[
                {
                    "role": "system",
                    "content": value,
                }
            ],
        )

        return completion.choices[0].message.content

    def vision_completion(
        self,
        last_prompt: str,
        image: Image,
        detail_level: ImageDetailLevel | None = None,
    ) -> str:
        content = [{"type": "text", "text": last_prompt}]

        if image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_base_64(image)}",
                        "detail": detail_level or "auto",
                    },
                }
            )

        completion = self._client.chat.completions.create(
            **self._default_params,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        return completion.choices[0].message.content

    def __call__(
        self, instruction: str, suffix: str = "", image: Image | None = None
    ) -> str:
        """
        Call the OpenAI LLM.

        Args:
            instruction (Prompt): A prompt object with instruction for LLM.
            suffix (str): Suffix to pass.

        Returns:
            str: Response
            :param image:
        """
        self.last_prompt = instruction + suffix

        if self.model in self._supported_chat_models:
            if image is not None:
                raise ValueError(
                    "Trying to pass image input to a model that does not support it"
                )
            response = self.chat_completion(self.last_prompt)
        elif self.model in self._supported_vision_models:
            response = self.vision_completion(self.last_prompt, image)
        else:
            raise ValueError("Unsupported model")

        return response

    @property
    def type(self) -> str:
        return "openai"
