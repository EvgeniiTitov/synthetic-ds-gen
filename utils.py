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

    return True


def get_class_names(logos_dir: str) -> List[str]:
    assert os.path.exists(logos_dir)
    class_names = list()
    for filename in os.listdir(logos_dir):
        filepath = os.path.join(logos_dir, filename)
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
            if filename.endswith(".txt"):
                continue
            image_path = os.path.join(path_to_dir, filename)
            if not os.path.splitext(filename)[-1].lower() in ALLOWED_EXTs:
                print(f"Not supported ext for image: {filename}, "
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


def split_backgrounds_between_workers(
        path_to_backgr: str,
        nb_of_workers: int
) -> List[str]:
    # Naive implementation, good for now
    paths = [
        os.path.join(path_to_backgr, e) for e in os.listdir(path_to_backgr)
    ]
    per_worker = len(paths) // nb_of_workers
    start = 0
    for i in range(1, nb_of_workers + 1):
        if i == nb_of_workers:
            yield paths[start:]
        else:
            yield paths[start: per_worker * i]
            start += per_worker


def create_train_val_dirs(dir_path: str) -> None:
    if not os.path.exists(os.path.join(dir_path, "train")):
        os.mkdir(os.path.join(dir_path, "train"))
    if not os.path.exists(os.path.join(dir_path, "valid")):
        os.mkdir(os.path.join(dir_path, "valid"))


def check_custom_params(logo_dir: str) -> dict:
    custom_params = dict()
    for folder in os.listdir(logo_dir):
        folder_path = os.path.join(logo_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                custom_params[folder] = dict()
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    for line in f:
                        # TODO: How to properly skip blank lines in between?
                        try:
                            k, v = line.rstrip().split("=")
                            k = k.strip()
                            if len(v.split()) > 1:
                                custom_params[folder][k] = [
                                    float(e) for e in v.split()
                                ]
                            else:
                                custom_params[folder][k] = float(v.split()[0])
                        except:
                            continue
                    print("==>Detected custom augmentation parameters for:",
                          folder)
    return custom_params


def get_default_params(params) -> dict:
    default = dict()
    default["nb_images"] = int(params["nb_images"])
    default["deform_limit"] = float(params["deform_limit"])
    default["deform_thresh"] = float(params["deform_thresh"])
    default["rotation_limit"] = int(params["rotation_limit"])
    default["rotation_thresh"] = float(params["rotation_thresh"])
    default["resize_limit"] = [
        float(e) for e in params["resize_limit"].split()
    ]
    default["noise_blur_thresh"] = float(params["noise_blur_thresh"])
    default["transp_range"] = [
        float(e) for e in params["transp_range"].split()
    ]
    default["transp_thresh"] = float(params["transp_thresh"])
    default["perspective_thresh"] = float(params["perspective_thresh"])
    default["perspective_range"] = [
        float(e) for e in params["perspective_range"].split()
    ]
    default["cutout_size"] = float(params["cutout_size"])
    default["cutout_nb"] = int(params["cutout_nb"])
    default["cutout_thresh"] = float(params["cutout_thresh"])
    default["color_thresh"] = float(params["color_thresh"])

    return default


def update_param_dict(default: dict, custom: dict) -> dict:
    for key, value in default.items():
        if key in custom.keys():
            default[key] = custom[key]

    return default


def format_and_validate_parameters(
        default_para,
        class_names: list,
        custom_para: dict = None,
) -> dict:
    default = get_default_params(default_para)
    params = dict()
    for class_name in class_names:
        if class_name not in custom_para.keys():
            params[class_name] = default
        else:
            params[class_name] = update_param_dict(default.copy(),
                                                   custom_para[class_name])
    assert len(params) == len(class_names)
    validate_params(params)

    return params


def validate_params(params: dict) -> None:
    for name, param in params.items():
        msg = f"Assertion error for class: {name}"
        assert 0 < int(param["nb_images"]) < 10_000, msg
        assert 0.0 <= float(param["deform_limit"]) < 1.0, msg
        assert 0.0 <= float(param["deform_thresh"]) <= 1.0, msg
        assert 0 <= int(param["rotation_limit"]) <= 180, msg
        assert 0.0 <= float(param["rotation_thresh"]) <= 1.0, msg
        assert all(
            [0.0 < float(e) < 0.2 for e in param["resize_limit"]]
        ), msg
        assert 0.0 <= float(param["noise_blur_thresh"]) <= 1.0, msg
        assert all(
            [0.0 < float(e) <= 1.0 for e in param["transp_range"]]
        ), msg
        assert 0.0 <= float(param["transp_thresh"]) <= 1.0, msg
        assert 0.0 <= float(param["perspective_thresh"]) <= 1.0, msg
        assert all(
            [0.0 <= float(e) < 0.15 for e in
             param["perspective_range"]]
        ), msg
        assert 0.0 <= float(param["cutout_size"]) < 1.0, msg
        assert 0 <= int(param["cutout_nb"]) < 6, msg
        assert 0.0 <= float(param["cutout_thresh"]) <= 1, msg
        assert 0.0 <= float(param["color_thresh"]) <= 1.0, msg
    print("Parameters validated")
