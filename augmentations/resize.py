import cv2
from typing import List
import numpy as np
import random


class Resize:
    """
    Resizes logo within the allowed range keeping the aspect ratio
    """
    def __init__(self, resize_range: List[float]):
        self.name = "resize"
        assert len(resize_range) == 2 and resize_range[0] < resize_range[1]
        assert all(0.0 < e < 1.0 for e in resize_range), "Wrong resize range values"
        self.min_allowed, self.max_allowed = resize_range

    def __call__(
            self,
            image: np.ndarray,
            background_size: List[int]
    ) -> np.ndarray:
        background_height, background_width = background_size
        image_height, image_width = image.shape[:2]
        resize_factor = random.randint(
            int(self.min_allowed * 100), int(self.max_allowed * 100)
        ) / 100
        # Take the longer side and rescale it according to the randomly picked
        # resize_factor, which is relative to the background image side
        if image.shape[0] > image.shape[1]:
            new_image_height = background_height * resize_factor
            aspect_ratio_factor = new_image_height / image_height
            new_image_width = image_width * aspect_ratio_factor
        else:
            new_image_width = background_width * resize_factor
            aspect_ratio_factor = new_image_width / image_width
            new_image_height = image_height * aspect_ratio_factor
        try:
            resized_image = cv2.resize(
                image,
                dsize=(int(new_image_width), int(new_image_height))
            )
        except Exception as e:
            print(f"Failed to resize logo. Error: {e}")
            raise e

        return resized_image
