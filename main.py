import os
import multiprocessing

import argparse
from typing import List
import cv2
import random

from augmentations import (
    Rotation, Resize, NoiseBlur, JPEGCompressor,
    Deformator, CutOut, PerspectiveWrapper, Color
)
from augmentations import Augmenter
from config import config
import utils


NEGATIVE_NAMING_FROM = 0


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
    parser.add_argument("--generating_negatives", type=int, choices=[0, 1],
                        default=0,
                        help="0: generating positives, save coordinates to txt"
                             "1: generating negatives, save empty txt files")
    parser.add_argument("--negative_thresh", type=float, default=0.2,
                        help="Thresh to apply a TN logo to a TN background")
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


def generate_positive_images(
        class_index: int,
        class_name: str,
        logo_dir: str,
        background_dir: str,
        nb_imgs_required: int,
        augmenter: Augmenter,
        img_count: int,
        save_path: str,
        split: float = None
) -> None:
    """
    Generates true positives by picking a random logo from the logo_dir and
    a random background from the background_dir. Applies augmentation towards
    the logo, overlays it on the background image and saves it alongside the
    coordinates (location on the background image where the logo was placed)
    """
    if split:
        save_train_path = os.path.join(save_path, "train")
        save_valid_path = os.path.join(save_path, "valid")
        valid_required = int(nb_imgs_required * split)
        train_required = nb_imgs_required - valid_required
        assert valid_required > 0 and train_required > 0

    background_gen = utils.get_background_image(background_dir)
    logs = dict()
    logo_paths = [os.path.join(logo_dir, e) for e in os.listdir(logo_dir)]
    total, exceptions = 0, 0
    while True:
        if total == nb_imgs_required:
            break

        # Read random logo and background image
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

        # Generate synthetic image
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

        # Save generated image
        if split:
            if total < train_required:
                store_path = os.path.join(save_train_path, f"{img_count}.jpg")
            else:
                store_path = os.path.join(save_valid_path, f"{img_count}.jpg")
        else:
            store_path = os.path.join(save_path, f"{img_count}.jpg")
        try:
            cv2.imwrite(store_path, image)
        except Exception:
            print(f"[ERROR]: Process: {os.getpid()} failed to write generate "
                  f"image on disk")
            exceptions += 1
            continue

        # Save txt file containing object coordinates
        if split:
            if total < train_required:
                txt_store_path = save_train_path
            else:
                txt_store_path = save_valid_path
        else:
            txt_store_path = save_path

        is_saved = utils.dump_coord_txt(
            cls=class_index, payload=coord, name=img_count,
            save_path=txt_store_path
        )
        if not is_saved:
            print(f"[ERROR]: Process: {os.getpid()} failed to save "
                  f"coordinates into txt")
            exceptions += 1
            continue

        # Keep track of augmentation applied alongside nb of images generated
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
        if split:
            logs_save_path = save_path
        else:
            logs_save_path = os.path.split(save_path)[0]

        utils.save_logs(logs, logs_save_path, class_name)
    print(f"Process: {os.getpid()} finishing with {exceptions} exceptions")


def generate_negative_images(
        logo_dir: str,
        background_paths: List[str],
        augmenter: Augmenter,
        img_count: int,
        save_path: str,
        logo_thresh: float = 0.2
) -> None:
    """
    Generate true negatives by picking a random logo from the logo_dir, a
    random background from the background dir, combines the two repeating the
    augmentation steps taken during true positive generation. Saves the image
    alongside an empty txt file.
    """
    print(f"Process: {os.getpid()} got {len(background_paths)} backgrounds")

    logo_paths = [os.path.join(logo_dir, e) for e in os.listdir(logo_dir)]
    exceptions = 0
    for i, backgr_path in enumerate(background_paths, start=1):
        # Read background (true negative) image
        backgr_image = cv2.imread(backgr_path)
        if backgr_image is None:
            print(f"[ERROR]: Process: {os.getpid()} failed to open "
                  f"negative: {backgr_path}")
            exceptions += 1
            continue

        # Apply true negative logo randomly
        image_to_save = None
        if random.random() > logo_thresh:
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
        store_path = os.path.join(save_path, f"{img_count}.jpg")
        try:
            cv2.imwrite(store_path, image_to_save)
        except Exception:
            print(f"[ERROR]: Process: {os.getpid()} failed to write generate "
                  f"image on disk")
            exceptions += 1
            continue

        is_saved = utils.dump_coord_txt(
            0, [], name=img_count, save_path=save_path
        )
        if not is_saved:
            print(f"[ERROR]: Process: {os.getpid()} failed to save "
                  f"an empty txt")
            exceptions += 1
            continue

        img_count += 1
        if i % 100 == 0:
            print(f"Process: {os.getpid()} generated {i} images")

    print(f"Process: {os.getpid()} finishing with {exceptions} exceptions")


def main():
    args = parse_arguments()
    cls_names = utils.get_class_names(args["logos"])

    # Create folders where generation results will be saved
    # Argument split determines saving format:
    # a) All generated images will be saved into 2 folders train and test
    # b) Each class gets its own folder in the save_path folder
    split = None
    if args["split"]:
        assert 0.0 < float(args["split"]) < 1.0
        split = args["split"]
        utils.create_train_val_dirs(args["save_path"])
    elif int(args["generating_negatives"]):
        pass
    else:
        created = utils.create_dest_dirs(args["save_path"], cls_names)
        if not created:
            return

    require_positives = False if int(args["generating_negatives"]) else True

    # Validate args to ensure correct parameters have been provided
    params = config["augmentation"]
    validate_parameters(params)

    # Validate provided logos are actually RGBA, else attempt converting
    exc = utils.validate_provided_logos(args["logos"], cls_names)
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
    color = Color(
        thresh=float(params["color_thresh"])
    )
    noise_blurer = NoiseBlur(
        types=["multiply", "contrast", "blur"],
        thresh=float(params["noise_blur_thresh"])
    )
    # NOTE: a) order matters; b) if tuple provided, only one augmentation in
    # the tuple will be selected and applied.
    logo_aug_before = [
        deformator,
        (color, noise_blurer),
        (perspective_wrapper, cutter),
        rotator,
        resizer
    ]
    logo_aug_after = [jpeg_compressor]

    # ----- ENTIRE FRAME AUGMENTATION -----
    pass

    augmenter = Augmenter(
        logo_aug_before=logo_aug_before,
        logo_aug_after=logo_aug_after,
        transp_thresh=float(params["transp_thresh"]),
        transp_range=[float(e) for e in params["transp_range"].split()]
    )

    print("\nSpawning workers")
    print("Cores available:", multiprocessing.cpu_count())
    processes = list()

    if require_positives:
        for i, cls_name in enumerate(cls_names):
            if split:
                save_path = args["save_path"]
            else:
                save_path = os.path.join(args["save_path"], cls_name)

            process = multiprocessing.Process(
                target=generate_positive_images,
                args=(
                    i,
                    cls_name,
                    os.path.join(args["logos"], cls_name),
                    args["background"],
                    int(params["nb_images"]),
                    augmenter,
                    int(params["nb_images"]) * i,
                    save_path,
                    split
                )
            )
            process.start()
            processes.append(process)
            print(f"Process {process.pid} spawned to generate: {cls_name}")
    else:
        background_splitter = utils.split_backgrounds_between_workers(
            path_to_backgr=args["background"],
            nb_of_workers=len(cls_names)
        )
        thresh = float(args["negative_thresh"])
        assert 0.0 <= thresh <= 1.0
        for i, cls_name in enumerate(cls_names):
            batch = next(background_splitter)
            process = multiprocessing.Process(
                target=generate_negative_images,
                args=(
                    os.path.join(args["logos"], cls_name),
                    batch,
                    augmenter,
                    NEGATIVE_NAMING_FROM + (len(batch) * i),
                    args["save_path"],
                    thresh
                )
            )
            process.start()
            processes.append(process)
            print(f"Process {process.pid} spawned to generate: {cls_name}")
    print(f"Total {len(processes)} processes spawned")

    for process in processes:
        process.join()
    print("Completed")


if __name__ == "__main__":
    main()
