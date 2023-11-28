import base64
from io import BytesIO
from typing import Literal

from pydantic import BaseModel
from PIL.Image import Image


class ImageEncoding(BaseModel):
    "Simple type to store images in JSON"

    type: str = "image"
    format: Literal["base64", "url"]
    content: str


def encode_image(image: Image) -> ImageEncoding:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return ImageEncoding(
        format="base64",
        content=img_str,
    )


class ImageBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Image: encode_image,
        }


class VQAInput(ImageBaseModel):
    image: Image
    question: str
