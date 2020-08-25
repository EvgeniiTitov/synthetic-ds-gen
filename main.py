import os
import sys

import argparse
import cv2
from typing import List
import numpy as np
import random

from augmentations import Rotation, Resize, NoiseBlur
from augmentations import Augmenter
from config import config


ALLOWED_EXTs = [".jpg", ".jpeg", ".png"]


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logos", required=True,
                        help="Dir with logos for augmentation")
    parser.add_argument("-b", "--background", required=True,
                        help="Dir with background images")
    parser.add_argument("-s", "--save_path", default=r"",
                        help="Dir where augmented images will be saved")
    arguments = parser.parse_args()

    return vars(arguments)


def validate_parameters(params) -> None:
    assert 0 < int(params["nb_images"]) < 10_000
    assert 0 <= int(params["rotation_limit"]) <= 180
    assert 0.0 <= float(params["rotation_thresh"]) <= 1.0
    assert 0.0 <= float(params["resize_thresh"]) <= 1.0
    assert all([0.0 < float(e) < 1.0 for e in params["resize_limit"].split()])
    assert 0.0 <= float(params["noise_blur_thresh"]) <= 1.0
    assert all([0.0 < float(e) <= 1.0 for e in params["transp_range"].split()])
    assert 0.0 <= float(params["transp_thresh"]) <= 1.0
    print("Parameters validated")


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

    print("Logos validated")
    return


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
            line = f"{cls} {' '.join(list(map(str, payload)))}"
            f.write(line)
    except Exception as e:
        print(f"Failed to create txt: {save_path}. Error: {e}")
        return False

    return True


def save_logs(payload: dict, save_path: str) -> None:
    filename = os.path.join(save_path, "aug_logs.txt")
    with open(filename, "w") as f:
        for img_name, aug in payload.items():
            line = f"{img_name} {' '.join(aug)}\n"
            f.write(line)

    return


def perform_augmentation(
        logos_dir: str,
        background_dir: str,
        save_path: str,
        cls_names: List[str],
        imgs_to_generate: int,
        augmenter: Augmenter
) -> None:
    """ Augmentation loop for each class """
    background_gen = get_background_image(background_dir)
    image_count, exceptions = 0, 0
    logs = dict()
    for i, cls_name in enumerate(cls_names):
        if cls_name != "rebel":
            continue

        logo_dir = os.path.join(logos_dir, cls_name)
        logo_paths = [
            os.path.join(logo_dir, e) for e in os.listdir(logo_dir)
        ]
        total = 0
        while True:
            if total == imgs_to_generate:
                break

            # Pick a random logo of class cls_name and a background image
            logo_path = random.choice(logo_paths)
            background_path = next(background_gen)
            logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            backgr_image = cv2.imread(background_path)
            if backgr_image is None:
                print(f"Failed to open background image: {background_path}")
                continue
            assert logo_image is not None
            # Generate an image, get coordinates of the place where the logo
            # was applied alongside the applied augmentation
            image, coord, augment = augmenter.generate_image(
                logo_image,
                backgr_image
            )
            if image is None:
                print("Generated image is None. Check the pipeline")
                exceptions += 1
                continue

            # cv2.rectangle(image, pt1=(coord[0], coord[1]),
            #               pt2=(coord[2], coord[3]),
            #               color=(0, 255, 0), thickness=3)
            # cv2.imshow("", image)
            # cv2.waitKey(0)
            # print(f"Class: {cls_name}. Img name: {image_count}.jpg. "
            #       f"Coords: {coord}. Augments: {augment}")

            store_path = os.path.join(
                save_path, cls_name, f"{image_count}.jpg"
            )
            try:
                cv2.imwrite(store_path, image)
            except Exception as e:
                print(f"Failed to save generated image. Error: {e}")
                exceptions += 1
                continue

            # Save coordinates next to the image
            dumped = dump_coord_txt(
                cls=i, payload=coord, name=image_count,
                save_path=os.path.join(save_path, cls_name)
            )
            if not dumped:
                print("Failed to dump coordinates into txt")
                inp = input("Continue? [Y/N]: ")
                if inp.strip().upper() == "N":
                    sys.exit()

            logs[f"{image_count}.jpg"] = augment
            total += 1
            image_count += 1
            print(f"Generated image: {total}, class: {cls_name}")

    save_logs(logs, save_path)
    return


def main():
    args = parse_arguments()

    # Validate args to ensure correct parameters have been provided
    params = config["augmentation"]
    validate_parameters(params)

    # Create folders where generation results will be saved
    cls_names = get_class_names(args["logos"])
    created = create_dest_dirs(args["save_path"], cls_names)
    if not created:
        return

    # Validate provided logos are actually RGBA, else attempt convertion
    validate_provided_logos(args["logos"], cls_names)

    # Initialize augmentators
    rotator = Rotation(
        rotation_limit=int(params["rotation_limit"]),
        rotation_thresh=float(params["rotation_thresh"])
    )
    resizer = Resize(
        resize_range=[float(e) for e in params["resize_limit"].split()],
        resize_thresh=float(params["resize_thresh"])
    )
    noise_blurer = NoiseBlur(
        types=["jpegcomp", "multiply", "contrast", "gaussian"],
        thresh=float(params["noise_blur_thresh"])
    )
    augmenter = Augmenter(
        logo_aug=[rotator, resizer],
        image_aug=[noise_blurer],
        transp_thresh=float(params["transp_thresh"]),
        transp_range=[float(e) for e in params["transp_range"].split()]
    )
    # Run image generation pipeline
    perform_augmentation(
        logos_dir=args["logos"],
        background_dir=args["background"],
        save_path=args["save_path"],
        cls_names=cls_names,
        imgs_to_generate=int(params["nb_images"]),
        augmenter=augmenter
    )


if __name__ == "__main__":
    main()
