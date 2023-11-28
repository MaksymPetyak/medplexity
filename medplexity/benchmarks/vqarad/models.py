from PIL.Image import Image

from medplexity.benchmarks.vqa_utils import ImageBaseModel


class VQARadEntry(ImageBaseModel):
    image: Image
    question: str
    answer: str

    class Config:
        arbitrary_types_allowed = True
