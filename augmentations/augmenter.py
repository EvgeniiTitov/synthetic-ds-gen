import numpy as np
import random
from typing import Tuple, List


class Augmenter:
    def __init__(
            self,
            logo_aug_before: list,
            image_aug: list,
            logo_aug_after: list,
            transp_thresh: float,
            transp_range: List[float]
    ):
        self.logo_transforms_before = logo_aug_before
        self.image_aug = image_aug
        self.logo_transforms_after = logo_aug_after
        self.trans_thresh = transp_thresh
        self.trans_min, self.trans_max = transp_range

    def generate_image(
            self,
            logo: np.ndarray,
            background: np.ndarray
    ) -> Tuple[np.ndarray, list, list]:
        """
        Applies transformations to the logo, overlays on top of the
        background and applies possible transformations to the entire image
        """
        log = list()
        backgr_size = background.shape[:2]

        # Apply transformations to the logo (resizing is compulsory step, its
        # thresh = 0.0) before it gets overlayed on top of background
        flag = True
        exclusive = ["perspective", "cutout"]
        for logo_t in self.logo_transforms_before:
            if random.random() > logo_t.thresh and logo_t.name not in exclusive:
                logo = logo_t(logo, background_size=backgr_size)
                log.append(logo_t.name)
            elif random.random() > logo_t.thresh and logo_t.name in exclusive and flag:
                logo = logo_t(logo, background_size=backgr_size)
                log.append(logo_t.name)
                flag = False

        # Combine the logo and background
        combined, coord_darknet, coord, transp_value = self._overlay_logo(
            logo, background
        )
        log.append(f"transp_value: {transp_value}")

        # Apply transformation to the entire image (blurring, noise etc)
        for image_t in self.image_aug:
            if random.random() > image_t.thresh:
                combined = image_t(combined)
                log.append(image_t.name)

        # Apply transformation to the image section on which the logo was
        # placed such as JPEG compression.
        x1, y1, x2, y2 = coord
        logo_arr = combined[y1:y2, x1:x2, :]
        for logo_post_t in self.logo_transforms_after:
            logo_arr = logo_post_t(logo_arr)
            log.append(logo_post_t.name)
        combined[y1:y2, x1:x2, :] = logo_arr

        return combined, coord_darknet, log

    def _overlay_logo(
            self,
            logo: np.ndarray,
            background: np.ndarray
    ) -> Tuple[np.ndarray, list, list, float]:
        """ Overlays company's logo on top of the provided background """
        logo_h, logo_w = logo.shape[:2]
        backgr_h, backgr_w = background.shape[:2]
        allowed_range_x = backgr_w - logo_w
        allowed_range_y = backgr_h - logo_h
        assert allowed_range_x > 0 and allowed_range_y > 0

        # Pick logo location coordinates
        x1 = random.randint(1, allowed_range_x)
        y1 = random.randint(1, allowed_range_y)
        x2 = x1 + logo_w
        y2 = y1 + logo_h
        assert x1 < x2 and y1 < y2 and all((x1 > 0, x2 > 0, y1 > 0, y2 > 0))
        assert x2 < backgr_w and y2 < backgr_h

        # Pick transparency value
        trans_factor = 1.0
        if random.random() > self.trans_thresh:
            trans_factor = float(random.randint(
                int(self.trans_min * 100), int(self.trans_max * 100)
            ) / 100.0)
            assert 0.0 < trans_factor <= 1.0

        overlay = logo[..., :3]
        mask = (logo[..., 3:] / 255.0) * trans_factor
        background[y1:y2, x1:x2] = (1.0 - mask) * \
                                    background[y1:y2, x1:x2] + mask * overlay

        coords = self.convert_coords_darknet_style(
            coords=[x1, y1, x2, y2],
            image=background
        )
        return background, coords, [x1, y1, x2, y2], trans_factor

    def convert_coords_darknet_style(
            self,
            coords: List[int],
            image: np.ndarray
    ) -> List[float]:
        """
        Converts coordinates from [left, top, right, bot] to the Darknet style:
        [x_centre, y_centre, width, height]
        """
        img_h, img_w = image.shape[:2]
        left, top, right, bot = coords

        box_centre_x = round(((left + right) // 2) / img_w, 6)
        box_centre_y = round(((top + bot) // 2) / img_h, 6)
        width = round((right - left) / img_w, 6)
        height = round((bot - top) / img_h, 6)

        assert all((0.0 <= e <= 1.0 for e in [box_centre_x, box_centre_y,
                                              width, height]))

        return [box_centre_x, box_centre_y, width, height]
