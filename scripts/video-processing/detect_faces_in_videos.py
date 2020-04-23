import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from src.align.align_trans import get_reference_facial_points, warp_and_crop_face
from src.align.detector import detect_faces
from src.align.get_nets import PNet, ONet, RNet

source_root = "../../data/fiw-mm/"
dest_root = "../../data/fiw-mm/interm/visual/video-frame-faces/"
crop_size = 112
fps = 25

path_data = Path(source_root).resolve()
path_images = path_data / 'FIDs-MM/visual/image'
path_videos = path_data / 'FIDs-MM/visual/video'
path_encodings = path_data / 'features/image/arcface'
path_out = Path(dest_root).resolve()
path_out.mkdir(exist_ok=True, parents=True)

scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

f_videos = list(path_videos.rglob('*.mp4'))
f_videos.sort()
f_videos = list(reversed(f_videos))
pnet = PNet()
rnet = RNet()
onet = ONet()

# for f_video in tqdm(f_videos):
for f_video in tqdm(f_videos):
    ref_base = str(f_video).replace(f"{str(path_videos)}/", "")
    path_obin = (path_out / ref_base).with_suffix('')
    try:
        path_obin.mkdir(parents=True)
    except:
        print('skipping', f_video)
        continue
    print("Processing\t{}".format(ref_base))
    clip = VideoFileClip(str(f_video))
    tracks = []
    for k, frame in enumerate(clip.iter_frames(fps=fps)):
        bbs, landmarks = detect_faces(Image.fromarray(frame), pnet=pnet, rnet=rnet, onet=onet)
        ndetections = len(landmarks)
        path_image_out = path_obin / 'frame-{:03d}.jpg'.format(k)
        if ndetections:  # If the landmarks cannot be detected, the img will be discarded
            for i in range(ndetections):
                facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(frame, facial5points, reference, crop_size=(crop_size, crop_size))

                path_image_out = path_obin / str(Path(path_image_out).name).replace('.jpg', '-{:02d}.jpg'.format(i))
                img_warped = Image.fromarray(warped_face)
                img_warped.save(str(path_image_out))

                path_lmarks_out = str(path_image_out).replace('.jpg', '-landmarks.csv')
                path_bb_out = str(path_image_out).replace('.jpg', '-bb.csv')

                np.savetxt(str(path_lmarks_out), landmarks[i], delimiter=',')
                np.savetxt(path_bb_out, bbs[i], delimiter=',')
        else:
            print("{} is discarded due to non-detected landmarks!".format(path_image_out))


def text(filenames):
    results = []
    for filename in filenames:
        basename = filename.split('.')[0]
        transcript = basename.replace('1', 'YES').replace('0', 'NO').replace('_', " ")
        results.append("{} {}".format(basename.split('.')[0], transcript))

    return '\n'.join(sorted(results))
