import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.align.align_trans import get_reference_facial_points, warp_and_crop_face
from src.align.detector import detect_faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument(
        "-source_root",
        "--source_root",
        help="specify your source dir",
        default="../../data/fiw-videos/new-processed/",
        type=str,
    )
    parser.add_argument(
        "-dest_root",
        "--dest_root",
        help="specify your destination dir",
        default="../../data/fiw-videos/new-processed/",
        type=str,
    )
    parser.add_argument(
        "-crop_size",
        "--crop_size",
        help="specify size of aligned faces, align and crop with padding",
        default=112,
        type=int,
    )
    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root  # specify your destination dir
    crop_size = (
        args.crop_size
    )  # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.0
    reference = get_reference_facial_points(default_square=True) * scale

    cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    dir_videos = glob.glob(f"{source_root}F????/v?????/")
    meta = {}
    # for subfolder in tqdm(os.listdir(source_root)):
    for subfolder in tqdm(reversed(dir_videos)):
        try:
            Path(subfolder).joinpath("scenes").joinpath("cropped").mkdir()
        except Exception:
            continue
        for imfile in (
            Path(subfolder).joinpath("scenes").joinpath("images").glob("*.jpg")
        ):
            print("Processing\t{}".format(imfile))
            img = Image.open(imfile)
            try:  # Handle exception
                bbs, landmarks = detect_faces(img)
            except Exception:
                print("{} is discarded due to exception!".format(imfile))
                continue
            ref = imfile._str.replace(source_root, "")
            ndetections = len(landmarks)
            if (
                ndetections == 0
            ):  # If the landmarks cannot be detected, the img will be discarded
                print("{} is discarded due to non-detected landmarks!".format(imfile))
                meta[ref] = []
                continue

            li_meta = []

            for i in range(ndetections):
                im_meta = {}
                im_meta["face"] = i
                im_meta["landmarks"] = landmarks[i]
                im_meta["bb"] = bbs[i]

                facial5points = [
                    [landmarks[i][j], landmarks[i][j + 5]] for j in range(5)
                ]
                warped_face = warp_and_crop_face(
                    np.array(img),
                    facial5points,
                    reference,
                    crop_size=(crop_size, crop_size),
                )
                img_warped = Image.fromarray(warped_face)
                image_name = imfile._str.replace("images", "cropped").replace(
                    ".jpg", "-{:02d}.jpg".format(i)
                )
                # im_meta['ref'] = "/".join(image_name.split('/')[-5:])
                img_warped.save(image_name)
                li_meta.append(im_meta)
            meta[ref] = li_meta
    with open(source_root + "cropped-meta.pkl", "wb") as f:
        pickle.dump(meta, f)
