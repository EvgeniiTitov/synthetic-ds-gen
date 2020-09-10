import imgaug.augmenters as iaa
import numpy as np


class Color:

    def __init__(self, thresh: float):
        self.name = "color"
        self.thresh = thresh
        self.transform = iaa.Add((-130, 130), per_channel=0.99)

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        img = image[:, :, :3]
        img = self.transform(image=img)
        image[:, :, :3] = img

        return image
