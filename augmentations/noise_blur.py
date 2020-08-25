import imgaug
from typing import List
import numpy as np


class NoiseBlur:
    """
    Applies noises to the entire image
    """
    def __init__(self, types: List[str], thresh: float):
        self.name = "noiseblur"
        self.thresh = thresh
        self.n = list()
        for type_ in types:
            if type_ == "jpegcomp":
                self.n.append(imgaug.augmenters.JpegCompression(
                    compression=(70, 100)
                ))
            elif type_ == "multiply":
                self.n.append(imgaug.augmenters.Multiply(
                    (0.5, 1.5), per_channel=0.5
                ))
            elif type_ == "contrast":
                self.n.append(imgaug.augmenters.GammaContrast(
                    gamma=(0.7, 1.7)
                ))
            elif type_ == "gaussian":
                self.n.append(imgaug.augmenters.GaussianBlur(
                    sigma=(0.2, 1.0)
                ))
            else:
                print("\n\nUnknown noise/blur type provided:", type_)

        if len(self.n) == 0:
            raise Exception(
                "Failed to initialize. No known noise/blur types provided"
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        aug = imgaug.augmenters.OneOf(self.n)
        return aug(image=image)
