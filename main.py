import os
import sys
import multiprocessing

import argparse
from typing import List, Tuple

from augmentations import (
    Rotation, Resize, NoiseBlur, JPEGCompressor,
    Deformator, CutOut, PerspectiveWrapper, Color
)
from augmentations import Augmenter
from config import config
from utils import *


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

    assert 0.0 <= float(params["deform_limit"]) < 1.0
    assert 0.0 <= float(params["deform_thresh"]) <= 1.0

    assert 0 <= int(params["rotation_limit"]) <= 180
    assert 0.0 <= float(params["rotation_thresh"]) <= 1.0

    assert all([0.0 < float(e) < 1.0 for e in params["resize_limit"].split()])

    assert 0.0 <= float(params["noise_blur_thresh"]) <= 1.0

    assert all([0.0 < float(e) <= 1.0 for e in params["transp_range"].split()])
    assert 0.0 <= float(params["transp_thresh"]) <= 1.0

    assert 0.0 <= float(params["perspective_thresh"]) <= 1.0
    assert all(
        [0.0 <= float(e) < 0.15 for e in params["perspective_range"].split()]
    )

    assert 0.0 <= float(params["cutout_size"]) < 1.0
    assert 0 <= int(params["cutout_nb"]) < 6
    assert 0.0 <= float(params["cutout_thresh"]) <= 1

    assert 0.0 <= float(params["color_thresh"]) <= 1.0

    print("Parameters validated")


def generate_images(
        class_index: int,
        class_name: str,
        logo_dir: str,
        background_dir: str,
        nb_imgs_required: int,
        augmenter: Augmenter,
        img_count: int,
        save_path: str
) -> None:
    """  """
    background_gen = get_background_image(background_dir)
    exceptions = 0
    logs = dict()
    logo_paths = [os.path.join(logo_dir, e) for e in os.listdir(logo_dir)]
    total = 0
    while True:
        if total == nb_imgs_required:
            break
        logo_path = random.choice(logo_paths)
        backgr_path = next(background_gen)
        logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        backgr_image = cv2.imread(backgr_path)
        if logo_image is None:
            print(f"[ERROR]: Process: {os.getpid()} failed to open "
                  f"logo: {logo_path}")
            exceptions += 1
            continue
        if backgr_image is None:
            print(f"[ERROR]: Process: {os.getpid()} failed to open "
                  f"background: {backgr_path}")
            exceptions += 1
            continue

        try:
            image, coord, augments = augmenter.generate_image(
                logo=logo_image,
                background=backgr_image
            )
        except Exception:
            exceptions += 1
            continue

        if image is None:
            print(f"[ERROR]: Img generated in process: {os.getpid()} is None")
            exceptions += 1
            continue

        store_path = os.path.join(save_path, f"{img_count}.jpg")
        try:
            cv2.imwrite(store_path, image)
        except Exception:
            print(f"[ERROR]: Process: {os.getpid()} failed to write generate "
                  f"image on disk")
            exceptions += 1
            continue

        is_saved = dump_coord_txt(
            cls=class_index, payload=coord, name=img_count,
            save_path=save_path
        )
        if not is_saved:
            print(f"[ERROR]: Process: {os.getpid()} failed to save "
                  f"coordinates into txt")
            exceptions += 1
            continue

        logs[f"{img_count}.jpg"] = augments
        total += 1
        img_count += 1

        if total % 100 == 0:
            print(f"Process: {os.getpid()} generated {total} images")

        # if exceptions == 10:
        #     print(f"Process {os.getpid()} terminated with"
        #           f" {exceptions} expections!")
        #     break
    if logs:
        save_logs(logs, os.path.split(save_path)[0], class_name)
    print(f"Process: {os.getpid()} finishing with {exceptions} exceptions")


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
    exc = validate_provided_logos(args["logos"], cls_names)
    print(f"Provided logos validated with {exc} warnings")

    # Initialize augmentators
    # ----- LOGO AUGMENTATORS -----
    deformator = Deformator(
        thresh=float(params["deform_thresh"]),
        deformation_limit=float(params["deform_limit"])
    )
    rotator = Rotation(
        rotation_limit=int(params["rotation_limit"]),
        rotation_thresh=float(params["rotation_thresh"])
    )
    resizer = Resize(
        resize_range=[float(e) for e in params["resize_limit"].split()]
    )
    cutter = CutOut(
        thresh=float(params["cutout_thresh"]),
        n=int(params["cutout_nb"]),
        size=float(params["cutout_size"]),
        squared=False
    )
    perspective_wrapper = PerspectiveWrapper(
        thresh=float(params["perspective_thresh"]),
        scale_limit=[float(e) for e in params["perspective_range"].split()]
    )
    jpeg_compressor = JPEGCompressor(thresh=0.01)
    color = Color(thresh=float(params["color_thresh"]))
    # NOTE - order matters
    logo_aug_before = [
        deformator, color, perspective_wrapper, cutter, rotator, resizer
    ]
    logo_aug_after = [jpeg_compressor]

    # ----- ENTIRE FRAME AUGMENTATION -----
    noise_blurer = NoiseBlur(
        types=["multiply", "contrast", "noise", "blur"],
        thresh=float(params["noise_blur_thresh"])
    )
    image_aug = [noise_blurer]

    augmenter = Augmenter(
        logo_aug_before=logo_aug_before,
        image_aug=image_aug,
        logo_aug_after=logo_aug_after,
        transp_thresh=float(params["transp_thresh"]),
        transp_range=[float(e) for e in params["transp_range"].split()]
    )

    print("\nSpawning workers")
    processes = list()
    for i, cls_name in enumerate(cls_names):
        process = multiprocessing.Process(
            target=generate_images,
            args=[i, cls_name,
                  os.path.join(args["logos"], cls_name),
                  args["background"],
                  int(params["nb_images"]),
                  augmenter,
                  int(params["nb_images"]) * i,
                  os.path.join(args["save_path"], cls_name)
            ]
        )
        process.start()
        processes.append(process)
        print(f"Process {process.pid} spawned to generate: {cls_name}")
    print(f"Total {len(processes)} processes spawned")

    for process in processes:
        process.join()
    print("Complete")


if __name__ == "__main__":
    main()
