import imgaug
import numpy as np


class JPEGCompressor:
    """
    Aplies JPEG compression on logo before it gets overlayed. 40, 80
    """
    def __init__(self, thresh: float):
        self.name = "jpegcomp"
        self.thresh = thresh
        self.transform = imgaug.augmenters.JpegCompression(
            compression=(10, 50)
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image=image)
