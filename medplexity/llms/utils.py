import base64
from io import BytesIO

from PIL.Image import Image


def encode_image_base_64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str
