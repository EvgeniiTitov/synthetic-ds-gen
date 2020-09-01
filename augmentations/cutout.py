import imgaug.augmenters as iaa
import numpy as np


class CutOut:

    def __init__(
            self,
            thresh: float,
            n: int = 3,
            size: float = 0.3,
            squared: bool = False
    ):
        self.name = "cutout"
        assert 0.0 <= thresh <= 1.0
        assert 0.0 <= size <= 1.0
        self.thresh = thresh
        self.transform = iaa.Cutout(
            nb_iterations=n, size=size, squared=squared, cval=0
        )

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self.transform(image=image)
