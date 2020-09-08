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
            scale=(scale_limit[0], scale_limit[1]), keep_size=True,
            fit_output=True
        )

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self.transform(image=image)


if __name__ == "__main__":
    import os
    import cv2
    t = PerspectiveWrapper(thresh=0.1, scale_limit=[0.05, 0.06])
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)

    imgs_path = r"D:\SingleView\logos\negatives\neg"
    for file in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, file)
        image = cv2.imread(img_path)
        if image is None:
            print("Failed to open image", file)

        image = t(image)
        cv2.imshow("window", image)

        if cv2.waitKey() == ord("q"):
            break
        elif cv2.waitKey(0):
            continue
