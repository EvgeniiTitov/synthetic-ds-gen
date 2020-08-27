import cv2
from typing import List
import numpy as np
import random
import math


class Resize:
    """
    Resizes logo within the allowed range keeping the aspect ratio
    """
    def __init__(self, resize_range: List[float]):
        self.name = "resize"
        assert len(resize_range) == 2 and resize_range[0] < resize_range[1]
        assert all(0.0 < e < 1.0 for e in resize_range), "Wrong resize range"
        self.min_allowed, self.max_allowed = resize_range
        assert self.min_allowed < self.max_allowed

    def __call__(
            self,
            image: np.ndarray,
            background_size: List[int]
    ) -> np.ndarray:
        background_height, background_width = background_size
        image_height, image_width = image.shape[:2]
        backgr_area = background_height * background_width

        # New desired ratio of logo area / background area
        dest_ratio = random.randint(
            int(self.min_allowed * 100), int(self.max_allowed * 100)
        ) / 100
        assert 0 < dest_ratio <= 1
        dest_area = int(backgr_area * dest_ratio)

        new_image_width = int(
            math.sqrt((dest_area * image_width) / (image_height))
        )
        new_image_height = int(
            (new_image_width * image_height) / image_width
        )
        assert new_image_width > 0 and new_image_height > 0
        assert new_image_height < background_height and \
                                            new_image_width < background_width
        try:
            resized_image = cv2.resize(
                image,
                dsize=(int(new_image_width), int(new_image_height))
            )
        except Exception as e:
            print(f"Failed to resize logo. Error: {e}")
            raise e

        return resized_image
