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


CORES_SATURATION_COEF = 1.25


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logos", required=True,
                        help="Dir with logos for augmentation")
    parser.add_argument("-b", "--background", required=True,
                        help="Dir with background images")
    parser.add_argument("-s", "--save_path", default=r"",
                        help="Dir where augmented images will be saved")
    parser.add_argument("--split", type=float, default=None,
                        help="Val split, implies different saving format")
    arguments = parser.parse_args()
    return vars(arguments)


def generate_positives(args: dict) -> None:
    pid = os.getpid()
    print(f"[INFO]: Process {pid} started")
    # Initialize augmentators
    deformator = Deformator(
        thresh=float(args["params"]["deform_thresh"]),
        deformation_limit=float(args["params"]["deform_limit"])
    )
    rotator = Rotation(
        rotation_limit=int(args["params"]["rotation_limit"]),
        rotation_thresh=float(args["params"]["rotation_thresh"]),
        rotation_type="bound"
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
    if args["split"]:
        save_train_path = os.path.join(args["save_path"], "train")
        save_valid_path = os.path.join(args["save_path"], "valid")
        valid_required = int(args["nb_imgs_required"] * args["split"])
        train_required = args["nb_imgs_required"] - valid_required
        assert valid_required > 0 and train_required > 0

    background_gen = utils.get_background_image(args["background_dir"])
    logs = dict()
    logo_paths = [
        os.path.join(args["logo_dir"], e) for e in
        os.listdir(args["logo_dir"]) if not e.endswith(".txt")
    ]
    total, exceptions = 0, 0
    img_count = args["img_count"]
    while True:
        if total == args["nb_imgs_required"]:
            break
        # Read random logo and background image
        logo_path = random.choice(logo_paths)
        backgr_path = next(background_gen)
        logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        backgr_image = cv2.imread(backgr_path)
        if logo_image is None:
            print(f"[ERROR]: Process: {pid} failed to open "
                  f"logo: {logo_path}")
            exceptions += 1
            continue
        if backgr_image is None:
            print(f"[ERROR]: Process: {pid} failed to open "
                  f"background: {backgr_path}")
            exceptions += 1
            continue

        # Generate synthetic image
        try:
            image, coord, augments = augmenter.generate_image(
                logo=logo_image,
                background=backgr_image
            )
        except Exception as e:
            exceptions += 1
            continue
        if image is None:
            print(f"[ERROR]: Img generated in process: {pid} is None")
            exceptions += 1
            continue

        # Save generated image
        if args["split"]:
            if total < train_required:
                store_path = os.path.join(save_train_path, f"{img_count}.jpg")
            else:
                store_path = os.path.join(save_valid_path, f"{img_count}.jpg")
        else:
            store_path = os.path.join(args["save_path"], f"{img_count}.jpg")
        try:
            cv2.imwrite(store_path, image)
        except Exception as e:
            print(
                f"[ERROR]: Process: {pid} failed to write generated "
                f"image on disk. Error: {e}")
            exceptions += 1
            continue

        # Save txt file containing object coordinates
        if args["split"]:
            if total < train_required:
                txt_store_path = save_train_path
            else:
                txt_store_path = save_valid_path
        else:
            txt_store_path = args["save_path"]
        is_saved = utils.dump_coord_txt(
            cls=args["class_index"], payload=coord,
            name=img_count, save_path=txt_store_path
        )
        if not is_saved:
            print(f"[ERROR]: Process: {pid} failed to save "
                  f"coordinates into txt")
            exceptions += 1
            continue

        # Keep track of augmentation applied alongside nb of images generated
        logs[f"{img_count}.jpg"] = augments
        total += 1
        img_count += 1
        if total % 100 == 0:
            print(f"[INFO]: Process {pid} generated {total} images")

    # Save augmentation logs
    if logs:
        if args["split"]:
            logs_save_path = args["save_path"]
        else:
            logs_save_path = os.path.split(args["save_path"])[0]
        utils.save_logs(logs, logs_save_path, args["class_name"])
    print(f"[INFO]: Process: {pid} finishing with {exceptions} exceptions")


def main():
    args = parse_arguments()
    cls_names = utils.get_class_names(args["logos"])

    # Create folders where generation results will be saved
    # Argument split determines saving format:
    # a) All generated images will be saved into 2 folders train and valid
    # b) Each class gets its own folder in the save_path folder
    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])
    split = None
    if args["split"]:
        assert 0.0 < float(args["split"]) < 1.0, "[ERROR]: Wrong split value"
        split = args["split"]
        utils.create_train_val_dirs(args["save_path"])
    else:
        created = utils.create_dest_dirs(args["save_path"], cls_names)
        if not created:
            return
        print("[INFO]: Destinations directories created")

    # Validate provided logos are actually RGBA, else attempt converting
    warn = utils.validate_provided_logos(args["logos"], cls_names)
    print(f"[INFO]: Provided logos validated with {warn} warnings")

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
    assert len(cls_names) == len(params), "[ERROR]: Mismatch"

    # Prepare args to distribute between workers
    to_distribute = list()
    arguments = {
        "class_index": None,
        "class_name": None,
        "logo_dir": None,
        "background_dir": args["background"],
        "nb_imgs_required": None,
        "img_count": None,
        "save_path": None,
        "params": None,
        "split": split
    }
    for key, (i, cls_name) in zip(params, enumerate(cls_names)):
        assert key == cls_name
        if split:
            save_path = args["save_path"]
        else:
            save_path = os.path.join(args["save_path"], cls_name)
        args_copy = arguments.copy()
        args_copy["class_index"] = i
        args_copy["class_name"] = cls_name
        args_copy["logo_dir"] = os.path.join(args["logos"], cls_name)
        args_copy["nb_imgs_required"] = int(params[key]["nb_images"])
        args_copy["img_count"] = int(params[key]["nb_images"]) * i
        args_copy["save_path"] = save_path
        args_copy["params"] = params[key]
        args_copy["split"] = split
        to_distribute.append(args_copy)

    cores = multiprocessing.cpu_count()
    nb_workers = int(cores * CORES_SATURATION_COEF)
    if nb_workers > len(cls_names):
        nb_workers = len(cls_names)
    print(f"\n[INFO]: Spawning {nb_workers} workers...")
    with multiprocessing.Pool(nb_workers) as p:
        p.map(generate_positives, tuple(to_distribute))

    print("[INFO]: Completed")


if __name__ == "__main__":
    main()
