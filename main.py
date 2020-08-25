import os

import argparse
import cv2
from typing import List
import numpy as np

from augmentations import Rotation, Resize, NoiseBlur, TransparencyOverlay


ALLOWED_EXTs = [".jpg", ".jpeg", ".png"]


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logos", required=True,
                        help="Dir with logos for augmentation")
    parser.add_argument("-b", "--background", required=True,
                        help="Dir with background images")
    parser.add_argument("-s", "--save_path", default=r"",
                        help="Dir where augmented images will be saved")
    parser.add_argument("--log", type=int,
                        help="1: create log file that lists what augmentation"
                             " was applied to each generated image")
    arguments = parser.parse_args()

    return vars(arguments)


def validate_args(args: dict) -> bool:
    return True


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
                print(f"Failed to create save dir for cls {cls_name}. Error: {e}")
                return False

    return True


def get_class_names(logos_dir: str) -> List[str]:
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
        print("Failed to open image")
        return False

    if image.shape[-1] != 4:
        return convert_to_rgba(image, image_path)
    else:
        return True


def validate_provided_logos(logos_dir: str, cls_names: List[str]) -> None:
    """
    Validate logos provided are suitable for synthetic augmentation. In
    case there's a not rgba image, attempt converting it.
    """
    print("\nValidating provided logos")
    for cls_name in cls_names:
        path_to_dir = os.path.join(logos_dir, cls_name)
        if not os.listdir(path_to_dir):
            raise Exception(f"No logos provided for the class: {cls_name}")

        # Validate each image in the folder
        for filename in os.listdir(path_to_dir):
            print("-" * 40)
            print(f"Validating image: {filename}, class: {cls_name}")
            image_path = os.path.join(path_to_dir, filename)
            if not os.path.splitext(filename)[-1].lower() in ALLOWED_EXTs:
                print(f"ATTENTION: Not allowed extension!")
                #os.remove(image_path)
                continue

            # Validate the image is suitable for augmentation
            confirmed = confirm_rgba(image_path)
            if not confirmed:
                print(f"Failed to validate image: {filename}, cls: {cls_name},"
                      f"deleted.")
                os.remove(image_path)
            else:
                print("Image validated successfully")

        assert len(os.listdir(path_to_dir)) > 0
    return


def perform_augmentation(
        logos_dir: str,
        save_path: str,
        cls_names: List[str],
        args: dict
) -> None:
    """ Augmentation loop for each class """
    rotator = Rotation(
        rotation_limit=,
        rotation_thresh=
    )
    resizer = Resize(
        resize_range=,
        resize_thresh=
    )
    noise_blurer = NoiseBlur(
        types=["jpegcomp", "multiply", "contrast", "gaussian"],
        thresh=
    )
    transparency_overlayer = TransparencyOverlay(
        transp_range=,
        transp_thresh=
    )



def main():
    args = parse_arguments()

    # Validate args to ensure correct parameters have been provided
    validated = validate_args(args)
    if not validated:
        return

    # Create folders where generation results will be saved
    cls_names = get_class_names(args["logos"])
    created = create_dest_dirs(args["save_path"], cls_names)
    if not created:
        return

    # Validate provided logos are actually RGBA, else attempt convertion
    validate_provided_logos(args["logos"], cls_names)
    perform_augmentation(args["logos"], args["save_path"], cls_names, args)


if __name__ == "__main__":
    main()
