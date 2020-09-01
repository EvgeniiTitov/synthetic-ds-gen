import random
import numpy as np
from typing import List
import imutils


class Rotation:
    """
    Random image rotation within the allowed range [-range:range]
    """
    def __init__(
            self,
            rotation_limit: int,
            rotation_thresh: float,
            rotation_type: str = "bound"
    ):
        self.name = "rotation"
        assert 0 <= rotation_limit <= 180, "Wrong range, expected: [0, 180]"
        self.range = rotation_limit
        self.thresh = rotation_thresh
        self.rotation_type = rotation_type

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        rotation_angle = random.randint(-self.range, self.range)
        if rotation_angle == 0:
            return image

        if self.rotation_type == "bound":
            rotated_image = imutils.rotate_bound(image, rotation_angle)
        else:
            rotated_image = imutils.rotate(image, rotation_angle)

        return rotated_image
