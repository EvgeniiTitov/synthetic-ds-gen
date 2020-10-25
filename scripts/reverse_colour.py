import os

import cv2

path_to_image = r"D:\SingleView\SpotIQ\logos\positives_channels\neon\2.png"
save_path = r"D:\SingleView\SpotIQ\logos\positives_channels\neon"

image = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)
assert image is not None
assert image.shape[-1] == 4, "Image is not transparent PNG"

alpha = image[:, :, -1]
rgb_image = image[:, :, :-1]

# Reverse colours for the actual image
rgb_image[rgb_image == 0] = 255

new_image = cv2.merge([rgb_image, alpha])

# cv2.imshow("", alpha)
# cv2.waitKey(0)
store_path = os.path.join(save_path, "reversed.png")
cv2.imwrite(store_path, new_image)
