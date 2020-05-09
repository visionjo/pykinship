import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from src.align.align_trans import get_reference_facial_points, warp_and_crop_face

# sys.path.append("../../")
from src.align.detector import detect_faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument(
        "-source_root",
        "--source_root",
        help="specify your source dir",
        # default="../../data/fiw-mm/",
        default="/home/jrobby/clips/",
        type=str,
    )
    parser.add_argument(
        "-dest_root",
        "--dest_root",
        help="specify your destination dir",
        # default="../../data/fiw-mm/features/visual/video/",
        default="/home/jrobby/clips-features/",
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

    path_data = Path(args.source_root).resolve()
    path_images = path_data / "FIDs-MM/visual/image"
    path_videos = path_data / "FIDs-MM/visual/video"
    path_encodings = path_data / "features/image/arcface"
    path_out = Path(args.dest_root)

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

    f_videos = path_videos.rglob("*.mp4")
    meta = {}
    for f_video in tqdm(f_videos):
        imfile = []
        ref = str(f_video).replace(str(path_data), "")
        print("Processing\t{}".format(f_video))
        clip = VideoFileClip(f_video)
        img = Image.open(imfile)
        try:  # Handle exception
            bbs, landmarks = detect_faces(img)
        except Exception:
            print("{} is discarded due to exception!".format(imfile))
            continue
        ref = imfile.replace(source_root, "")
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

            facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(
                np.array(img),
                facial5points,
                reference,
                crop_size=(crop_size, crop_size),
            )
            img_warped = Image.fromarray(warped_face)
            image_name = imfile.replace("images", "cropped").replace(
                ".jpg", "-{:02d}.jpg".format(i)
            )
            # im_meta['ref'] = "/".join(image_name.split('/')[-5:])
            img_warped.save(image_name)
            li_meta.append(im_meta)
        meta[ref] = li_meta
with open(source_root + "cropped-meta.pkl", "wb") as f:
    pickle.dump(meta, f)
