import numpy as np
import random

class Augmenter:
    def __init__(self, augmentations: list):
        self.aug = augmentations

    def generate_image(
            self,
            logo: np.ndarray,
            background: np.ndarray
    ) -> np.ndarray:
        pass
