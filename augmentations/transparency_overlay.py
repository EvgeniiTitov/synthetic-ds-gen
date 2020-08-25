import random
import numpy as np
from typing import List


class TransparencyOverlay:
    """
    Makes the overlay image transparent and applies it to the background one
    """
    def __init__(self, transp_range: List[float], transp_thresh: float):
        self.name = "transparency"
        assert len(transp_range) == 2 and transp_range[0] < transp_range[1]
        assert all(0.0 < e < 1.0 for e in transp_range), "Wrong range provided"
        self.transp_min, self.transp_max = transp_range
        self.thresh = transp_thresh

    def __call__(
            self,
            background: np.ndarray,
            overlay: np.ndarray,
            x: int, y: int
    ) -> np.ndarray:
        transparency_factor = float(random.randint(
            int(self.transp_min * 100),
            int(self.transp_max * 100)
        ) / 100)
        assert 0.0 <= transparency_factor <= 1.0
        background_width = background.shape[1]
        background_height = background.shape[0]
        # Check if x, y coordinates are beyond the background image edges
        if x >= background_width or y >= background_height:
            return background

        # If overlay image goes beyond background image edges, cut the array
        # accordingly to keep only the overlay image area that happens to be
        # within the background image
        h, w = overlay.shape[0], overlay.shape[1]
        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]
        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        overlay_image = overlay[..., :3]  # rgb image
        mask = (overlay[..., 3:] / 255.0) * transparency_factor
        background[y:y + h, x:x + w] = (1.0 - mask) * \
                            background[y:y + h, x:x + w] + mask * overlay_image

        return background
