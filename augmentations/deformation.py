import numpy as np
import random
from typing import List
import cv2


class Deformator:
    WIDTH_THRESH = 4

    def __init__(self, thresh: float, deformation_limit: float = 0.2):
        self.name = "deformation"
        assert 0.0 <= thresh <= 1.0
        assert 0.0 <= deformation_limit < 1.0
        self.thresh = thresh
        self.limit = deformation_limit

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Deforms logo (stretches/squishes) logo along either X or Y axis.
        """
        img_h, img_w = image.shape[:2]
        # Do not make already wide short logos any wider or shorter
        if img_w / img_h > self.WIDTH_THRESH:
            dim = random.randint(0, 1)
            if dim == 0:
                deform = random.randrange(
                    100, int(100 + self.limit * 100)
                ) / 100.0
                new_size = (img_w, int(img_h * deform))
            else:
                deform = random.randrange(
                    int(100 - self.limit * 100), 100
                ) / 100.0
                new_size = (int(img_w * deform), img_h)
        else:
            deform = random.randrange(
                int(100 - self.limit * 100), int(100 + self.limit * 100)
            ) / 100.0
            dim = random.randint(0, 1)
            if dim == 0:
                new_size = (img_w, int(img_h * deform))
            else:
                new_size = (int(img_w * deform), img_h)

        return cv2.resize(image, dsize=new_size)
