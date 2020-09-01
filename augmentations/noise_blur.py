import imgaug
from typing import List
import numpy as np


class NoiseBlur:
    """
    Applies noises to the entire image
    """
    def __init__(self, types: List[str], thresh: float):
        self.name = "noiseblur"
        self.thresh = thresh
        self.n = list()
        for type_ in types:
            if type_ == "multiply":
                self.n.append(imgaug.augmenters.Multiply(
                    (0.3, 1.6), per_channel=0.9
                ))
            elif type_ == "contrast":
                self.n.append(imgaug.augmenters.GammaContrast(
                    gamma=(0.5, 2.0)
                ))
            elif type_ == "blur":
                self.n.append(imgaug.augmenters.GaussianBlur(
                    sigma=(0.2, 1.0)
                ))
            elif type_ == "noise":
                self.n.append(imgaug.augmenters.imgcorruptlike.GaussianNoise(
                    severity=1)
                )
            else:
                print("\n\nUnknown noise/blur type provided:", type_)

        if len(self.n) == 0:
            raise Exception(
                "Failed to initialize. No known noise/blur types provided"
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        aug = imgaug.augmenters.OneOf(self.n)
        return aug(image=image)


if __name__ == "__main__":
    import cv2
    img_path = r"D:\Desktop\system_output\SpotIQ_test\logos\countdown\1.png"
    img = cv2.imread(img_path)
    assert img is not None
    trans = NoiseBlur(["noise"], 0.3)

    for i in range(50):
        image = trans(img)
        cv2.imshow("", image)
        cv2.waitKey(0)
