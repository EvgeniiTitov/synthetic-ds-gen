import numpy as np
import imgaug.augmenters as iaa


class PerspectiveWrapper:
    """
    Applies affine (perspective) transforms
    """
    def __init__(self, thresh: float, scale_limit: list):
        self.name = "perspective"
        assert 0.0 <= thresh <= 1.0
        self.thresh = thresh
        self.transform = iaa.PerspectiveTransform(
            scale=(scale_limit[0], scale_limit[1])
        )

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self.transform(image=image)
