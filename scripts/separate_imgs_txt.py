# As per Ultralytics requirements 
import os 
import argparse


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True,
                        help="Dir with image/txt pairs")
    parser.add_argument("-s", "--save_path", 
                        help="Dir where folders imgs and txts will be created")
    arguments = parser.parse_args()
    return vars(arguments)


def create_dest_folders(dir_: str) -> tuple:
    images_dir = os.path.join(dir_, "images")
    txts_dir = os.path.join(dir_, "txts")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(txts_dir):
        os.mkdir(txts_dir)
        
    return images_dir, txts_dir


def main() -> None:
    args = parse_arguments()
    if args["save_path"]:
        if not os.path.exists(args["save_path"]):
            os.mkdir(args["save_path"])
        imgs_dir, txt_dir = create_dest_folders(args["save_path"])
    else:
        imgs_dir, txt_dir = create_dest_folders(args["dir"])
    
    for filename in os.listdir(args["dir"]):
        filepath = os.path.join(args["dir"], filename)
        if not os.path.isfile(filepath):
            continue
        if filename.endswith(".txt"):
            os.rename(filepath, os.path.join(txt_dir, filename))
        else:
            os.rename(filepath, os.path.join(imgs_dir, filename))
        print("Moved:", filename)

if __name__ == "__main__":
    main()
