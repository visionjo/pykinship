import glob
import sys
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append('../')
from src.data.videos import crop_detection


def process_scenes(dir_in, dir_out):
    """ process first, middle, and last image of each scene to determine whether MID is present.
    :param
    """
    Path(dir_out).mkdir(exist_ok=True)
    imfiles = glob.glob(dir_in + '*.jpg')
    imfiles.sort()
    # imfiles = np.reshape(imfiles[:-1], (-1, 3))
    for i, imfile in enumerate(imfiles):
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # for face_id, imfile in enumerate(imstack):
        suffix = imfile.split('.')[-2][-7:]
        # if Path(dir_out).joinpath("s{}-{:02d}.png".format(suffix, 0)).is_file():
        #     continue

        frame = cv2.imread(imfile)
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

        for j, (top, right, bottom, left) in enumerate(face_locations):
            # Draw a box around the face
            imout = "s{}-{:02d}.png".format(suffix, j)
            print(imout)
            face_locations_dict = {
                "frame": suffix.split('-')[1],
                "face": j,
                "bb": (left, top, right, bottom),
                "landmarks": face_locations[j],
                "path": imout
            }

            face_image = crop_detection(Image.fromarray(rgb_frame), face_locations_dict)
            feature = face_recognition.face_encodings(np.array(face_image))
            if not len(feature):
                continue
            feature = feature[0]

            face_image.save(dir_out + imout)
            pd.to_pickle(
                feature, f"{dir_out}s{suffix}-{j}-encoding.pkl"
            )
            pd.DataFrame().from_dict(face_locations_dict.items()).T.to_json(
                f"{dir_out}s{suffix}-{j}-meta.json"
            )


# dir_data = '../data/fiw-videos/new-processed/'
dir_data = '/home/jrobby/new-processed/'
dirs_scenes = glob.glob(dir_data + '*/*/scenes/')
dirs_scenes.sort()

# vids = [d.replace(dir_data, '').replace('/scenes/', '') for d in dirs_scenes]

for dir_scene in tqdm(dirs_scenes):
    dout = dir_scene + '/faces/'
    din = dir_scene + '/images/'
    if Path(dout).exists():
        continue
    process_scenes(din, dout)
