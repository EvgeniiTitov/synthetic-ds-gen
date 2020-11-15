import os
import multiprocessing

import argparse
import cv2
import random

from augmentations import (
    Rotation, Resize, NoiseBlur, JPEGCompressor,
    Deformator, CutOut, PerspectiveWrapper, Color
)
from augmentations import Augmenter
from config import config
import utils


NEGATIVE_NAMING_FROM = 655_000
CORES_SATURATION_COEF = 1.25


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logos", required=True,
                        help="Dir with logos for augmentation")
    parser.add_argument("-b", "--background", required=True,
                        help="Dir with background images")
    parser.add_argument("-s", "--save_path", default=r"",
                        help="Dir where augmented images will be saved")
    parser.add_argument("--negative_thresh", type=float, default=0.6,
                        help="Thresh to apply a TN logo to a TN background")
    arguments = parser.parse_args()
    return vars(arguments)


def generate_negatives(args: dict) -> None:
    pid = os.getpid()
    print(f"Process: {pid} started")

    # Initialize augmentators
    deformator = Deformator(
        thresh=float(args["params"]["deform_thresh"]),
        deformation_limit=float(args["params"]["deform_limit"])
    )
    rotator = Rotation(
        rotation_limit=int(args["params"]["rotation_limit"]),
        rotation_thresh=float(args["params"]["rotation_thresh"])
    )
    resizer = Resize(
        resize_range=[float(e) for e in args["params"]["resize_limit"]]
    )
    cutter = CutOut(
        thresh=float(args["params"]["cutout_thresh"]),
        n=int(args["params"]["cutout_nb"]),
        size=float(args["params"]["cutout_size"]),
        squared=False
    )
    perspective_wrapper = PerspectiveWrapper(
        thresh=float(args["params"]["perspective_thresh"]),
        scale_limit=[float(e) for e in args["params"]["perspective_range"]]
    )
    jpeg_compressor = JPEGCompressor(thresh=0.01)
    color = Color(
        thresh=float(args["params"]["color_thresh"])
    )
    noise_blurer = NoiseBlur(
        types=["multiply", "contrast", "blur"],
        thresh=float(args["params"]["noise_blur_thresh"])
    )
    # NOTE: a) order matters; b) if tuple provided, only one augmentation in
    # the tuple will be randomly selected and applied.
    logo_aug_before = [
        deformator,
        (color, noise_blurer),
        (perspective_wrapper, cutter),
        rotator,
        resizer
    ]
    logo_aug_after = [jpeg_compressor]
    augmenter = Augmenter(
        logo_aug_before=logo_aug_before,
        logo_aug_after=logo_aug_after,
        transp_thresh=float(args["params"]["transp_thresh"]),
        transp_range=[float(e) for e in args["params"]["transp_range"]]
    )
    logo_paths = [
        os.path.join(args["logo_dir"], e)
        for e in os.listdir(args["logo_dir"]) if not e.endswith(".txt")
    ]
    exceptions = 0
    img_count = args["img_count"]
    for i, backgr_path in enumerate(args["background_paths"], start=1):
        # Read background (true negative) image
        backgr_image = cv2.imread(backgr_path)
        if backgr_image is None:
            print(f"[ERROR]: Process {pid} failed to open "
                  f"background: {backgr_path}")
            exceptions += 1
            continue

        # Apply true negative logo to a random location
        if random.random() > args["logo_thresh"]:
            logo_path = random.choice(logo_paths)
            logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_image is not None:
                # Generate synthetic image
                try:
                    gen_image, coord, augments = augmenter.generate_image(
                        logo=logo_image,
                        background=backgr_image
                    )
                    image_to_save = gen_image
                except Exception:
                    exceptions += 1
                    image_to_save = backgr_image
            else:
                image_to_save = backgr_image
        else:
            image_to_save = backgr_image

        # Save image
        store_path = os.path.join(args["save_path"], f"{img_count}.jpg")
        try:
            cv2.imwrite(store_path, image_to_save)
        except Exception:
            print(f"[ERROR]: Process: {pid} failed to write generated"
                  f" image on disk")
            exceptions += 1
            continue

        is_saved = utils.dump_coord_txt(
            0, [], name=img_count, save_path=args["save_path"]
        )
        if not is_saved:
            print(f"[ERROR]: Process: {pid} failed to save an empty txt")
            exceptions += 1
            continue

        img_count += 1
        if i % 100 == 0:
            print(f"Process: {pid} generated {i} images")
    print(f"[INFO]: Process: {pid} finishing with {exceptions} exceptions")


def main():
    args = parse_arguments()

    # Get list of class names
    cls_names = utils.get_class_names(args["logos"])

    # Create folders where generation results will be saved
    # Argument split determines saving format:
    # a) All generated images will be saved into 2 folders train and valid
    # b) Each class gets its own folder in the save_path folder
    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])

    # Validate provided logos are actually RGBA, else attempt converting
    warnings = utils.validate_provided_logos(args["logos"], cls_names)
    print(f"[INFO]: Provided logos validated with {warnings} warnings")

    # Get default augmentation parameters (applies to all classes),
    # and check for any custom ones if any (in case default not suitable for
    # some class and it requires custom parameters). Validate afterwards
    params = config["augmentation"]
    custom_params = utils.check_custom_params(args["logos"])
    params = utils.format_and_validate_parameters(
        default_para=params,
        custom_para=custom_params,
        class_names=cls_names
    )
    assert len(cls_names) == len(params)

    # Prepare args to distribute between workers
    to_distribute = list()
    arguments = {
        "logo_dir": None,
        "background_paths": None,
        "params": None,
        "img_count": None,
        "save_path": None,
        "logo_thresh": None
    }
    background_splitter = utils.split_backgrounds_between_workers(
        path_to_backgr=args["background"],
        nb_of_workers=len(cls_names)
    )
    thresh = float(args["negative_thresh"])
    assert 0.0 <= thresh <= 1.0, "[ERROR]: Wrong threshold value"
    for key, (i, cls_name) in zip(params, enumerate(cls_names)):
        assert key == cls_name
        batch = next(background_splitter)
        print("Per batch:", len(batch))
        args_copy = arguments.copy()
        args_copy["logo_dir"] = os.path.join(args["logos"], cls_name)
        args_copy["background_paths"] = batch
        args_copy["params"] = params[key]
        args_copy["img_count"] = NEGATIVE_NAMING_FROM + (len(batch) * i)
        args_copy["save_path"] = args["save_path"]
        args_copy["logo_thresh"] = thresh
        to_distribute.append(args_copy)

    cores = multiprocessing.cpu_count()
    nb_workers = int(cores * CORES_SATURATION_COEF)
    if nb_workers > len(cls_names):
        nb_workers = len(cls_names)
    print(f"\nSpawning {nb_workers} workers")
    with multiprocessing.Pool(nb_workers) as p:
        p.map(generate_negatives, tuple(to_distribute))

    print("Completed")


if __name__ == "__main__":
    main()
