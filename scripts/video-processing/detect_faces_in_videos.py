""""
Detect faces in videos (set)

Save JPG (cropped faces) and CSVs (BB and landmarks)

TODO - optimize for batch processing and to run using cuda
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from moviepy.editor import VideoFileClip
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PACKAGE_PARENT = "../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
print(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import sys

sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.align.align_trans import get_reference_facial_points, warp_and_crop_face
from src.align.detector import detect_faces
from src.align.get_nets import PNet, ONet, RNet

cuda, Tensor = (
    (True, torch.cuda.FloatTensor)
    if torch.cuda.is_available()
    else (False, torch.FloatTensor)
)


def text(filenames):
    results = []
    for filename in filenames:
        basename = filename.split(".")[0]
        transcript = basename.replace("1", "YES").replace("0", "NO").replace("_", " ")
        results.append("{} {}".format(basename.split(".")[0], transcript))

    return "\n".join(sorted(results))


# source_root = Path("/Volumes/MyWorld/FIW-MM/clips")
# dest_root = "/Volumes/MyWorld/FIW-MM/clips-faces"
crop_size = 112
fps = 25
imsize = (112, 112)

source_root = Path("/home/jrobby/VIDs-aligned-tp-faces/")
dest_root = Path("/home/jrobby/VIDs-aligned-tp-faces-aligned/")
model_root = "../../models/Backbone_IR_152_checkpoint.pth"
# path_data = Path(source_root).resolve()ß

path_out = Path(dest_root).resolve()
path_out.mkdir(exist_ok=True, parents=True)

scale = crop_size / 112.0
reference = get_reference_facial_points(default_square=True) * scale

dir_frames = list(source_root.glob("F????/MID*/v?????/track-*"))
dir_frames.sort()
# dir_frames = list(reversed(dir_frames))

pnet = PNet()
rnet = RNet()
onet = ONet()

if cuda:
    pnet.cuda()
    rnet.cuda()
    onet.cuda()

for dir_fid in tqdm(reversed(dir_frames)):

    for f_video in dir_fid.glob("*.jpg"):
        try:
            ref_base = (
                str(f_video).replace(f"{str(source_root)}/", "").replace(".jpg", "")
            )
            path_obin = (path_out / ref_base).with_suffix("")
            try:
                path_obin.mkdir(parents=True)

            except Exception as e:
                print("skipping", f_video, e)
                continue
            print("Processing\t{}".format(ref_base))

            clip = VideoFileClip(str(f_video))

            tracks = []

            for k, frame in enumerate(clip.iter_frames(fps=fps)):
                try:
                    bbs, landmarks = detect_faces(
                        Image.fromarray(frame), pnet=pnet, rnet=rnet, onet=onet
                    )

                    ndetections = len(landmarks)
                    path_image_out = path_obin / "frame-{:03d}.jpg".format(k)
                    if ndetections:
                        # If the landmarks cannot be detected, the img will be discarded
                        for i in range(ndetections):
                            facial5points = [
                                [landmarks[i][j], landmarks[i][j + 5]] for j in range(5)
                            ]
                            warped_face = warp_and_crop_face(
                                frame,
                                facial5points,
                                reference,
                                crop_size=(crop_size, crop_size),
                            )

                            path_image_out = path_obin / str(
                                Path(path_image_out).name
                            ).replace(".jpg", "-{:02d}.jpg".format(i))
                            img_warped = Image.fromarray(warped_face)
                            img_warped.save(str(path_image_out))

                            path_lmarks_out = str(path_image_out).replace(
                                ".jpg", "-landmarks.csv"
                            )
                            path_bb_out = str(path_image_out).replace(".jpg", "-bb.csv")

                            np.savetxt(
                                str(path_lmarks_out), landmarks[i], delimiter=","
                            )
                            np.savetxt(path_bb_out, bbs[i], delimiter=",")

                    else:
                        print(
                            "{} is discarded due to non-detected landmarks!".format(
                                path_image_out
                            )
                        )
                except Exception as e:
                    print(e)
            # os.remove(str(f_video))
        except Exception as e:
            print("corrupted", f_video)
