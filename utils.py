import os

import cv2
from typing import List
import numpy as np
import random


ALLOWED_EXTs = [".jpg", ".jpeg", ".png"]


def create_dest_dirs(save_path: str, cls_names: List[str]) -> bool:
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except Exception as e:
            print(f"Failed to create the dir to save results in. Error: {e}")
            return False

    for cls_name in cls_names:
        dirname = os.path.join(save_path, cls_name)
        if not os.path.exists(dirname):
            try:
                os.mkdir(dirname)
            except Exception as e:
                print(f"Failed to create save dir for {cls_name}. Error: {e}")
                return False

    print("Destinations directories created")
    return True


def get_class_names(logos_dir: str) -> List[str]:
    assert os.path.exists(logos_dir)
    class_names = list()
    for filename in os.listdir(logos_dir):
        filepath = os.path.join(filename, logos_dir)
        if os.path.isdir(filepath):
            class_names.append(filename)
        else:
            print("INFO: not dir object in the logo folder!")

    return class_names


def convert_to_rgba(image: np.ndarray, image_path: str) -> bool:
    # Convert image to RGBA
    b, g, r = cv2.split(image)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    image = cv2.merge((b, g, r, alpha))

    # Save converted image next to the original one
    path, image_name = os.path.split(image_path)
    new_name = os.path.splitext(image_name)[0] + "_converted.png"
    save_path = os.path.join(path, new_name)
    try:
        cv2.imwrite(save_path, image)
    except Exception as e:
        print(f"Failed to save converted to RGBA image named: {image_name}. "
              f"Error: {e}")
        return False

    # Validate converted image by opening it and checking number of channels
    img = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.shape[-1] != 4:
        return False
    else:
        print("Image converted to RGBA. Deleting the old one")
        os.remove(image_path)
        return True


def confirm_rgba(image_path: str) -> bool:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        return False

    if image.shape[-1] != 4:
        return convert_to_rgba(image, image_path)
    else:
        return True


def validate_provided_logos(logos_dir: str, cls_names: List[str]) -> int:
    """
    Validate logos provided are suitable for synthetic augmentation. In
    case there's a not rgba image, attempt converting it.
    """
    print("\nValidating provided logos")
    warning = 0
    for cls_name in cls_names:
        path_to_dir = os.path.join(logos_dir, cls_name)
        if not os.listdir(path_to_dir):
            raise Exception(f"No logos provided for the class: {cls_name}")

        # Validate each image in the folder
        for filename in os.listdir(path_to_dir):
            image_path = os.path.join(path_to_dir, filename)
            if not os.path.splitext(filename)[-1].lower() in ALLOWED_EXTs:
                print(f"\nNot supported ext for image: {filename}, "
                      f"class: {cls_name}!")
                warning += 1
                continue

            # Validate the image is suitable for augmentation
            confirmed = confirm_rgba(image_path)
            if not confirmed:
                print(f"\nFailed to convert image: {filename}, cls: {cls_name}"
                      f"to RGBA.")
                #os.remove(image_path)
                warning += 1
        assert len(os.listdir(path_to_dir)) > 0

    return warning


def get_background_image(background_dir: str) -> str:
    assert len(os.listdir(background_dir)) > 0, "No background images provided"
    image_names = os.listdir(background_dir)
    while True:
        image_name = random.choice(image_names)
        yield os.path.join(background_dir, image_name)


def dump_coord_txt(cls: int, payload: list, name: int, save_path: str) -> bool:
    filename = os.path.join(save_path, str(name) + ".txt")
    try:
        with open(filename, "w") as f:
            if payload:
                line = f"{cls} {' '.join(list(map(str, payload)))}"
                f.write(line)
    except Exception as e:
        print(f"Failed to create txt: {save_path}. Error: {e}")
        return False

    return True


def save_logs(payload: dict, save_path: str, name: str = None) -> None:
    if name:
        filename = os.path.join(save_path, f"{name}_logs.txt")
    else:
        filename = os.path.join(save_path, "aug_logs.txt")
    with open(filename, "w") as f:
        for img_name, aug in payload.items():
            line = f"{img_name}: {' '.join(aug)}\n"
            f.write(line)

    return
