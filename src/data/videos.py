# import libraries
import glob
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image

fetch_videos = False


def wget_video(
        name, url, cmd="youtube-dl --continue --merge-output-format mp4 --all-subs -o {} {}"
):
    """
    Fetch video from youtube
    :param name: identifier to name video with
    :param url: youtube URL to download as MP4
    :param cmd: command line call (Note, str.format() assumes 2 placeholders
    :return: True if successfully downloaded; else, return False.
    """

    try:

        os.system(cmd.format(name, url))
        return True
    finally:
        return False


def encode_face_files(imfiles):
    images = OrderedDict({impath: cv2.imread(impath)[:, :, ::-1] for impath in imfiles})

    return {
        path: face_recognition.face_encodings(images[path])[0] for path in images.keys()
    }


def fetch_videos(f_csv):
    df = pd.read_csv(f_csv)
    df["ref"] = df["firstname"] + "_" + df["surname"]
    df.apply(lambda x: wget_video(x["ref"], x["video"]), axis=1)


def encode_mids(f_video=None, f_encodings=None, save_pickle=True):
    if Path(f_encodings).is_file():
        encodings = pd.read_pickle(f_encodings)
    else:
        impaths = glob.glob(f"{str(Path(f_video).parent)}/*.jpg")
        encodings = encode_face_files(impaths)
        if save_pickle:
            pd.to_pickle(encodings, encodings)
    return encodings


def process_subject(encodings, f_video, dir_out):
    video_capture = cv2.VideoCapture(f_video)
    # impaths = Path(dir_mid).glob('*.jpg')

    while True:

        # Grab a single frame of video
        ret, frame = video_capture.read()
        # if Path(f"{dout}fr{frame_id}_face{0}.png").exists():
        #     # face_landmarks_list = face_recognition.face_landmarks(frame[:, :, ::-1])
        #     # pd.DataFrame().from_dict(face_landmarks_list[0].items()).T.to_json(f"{dout}fr{frame_id}_face{0}-landmarks.json")
        #     frame_id += 1
        #     continue
        print(Path(f"fr{frame_id}_face{0}.png"))
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

        # Initialize variables
        # face_locations_all = []
        frame_id = 0
        # Display the results
        for j, (top, right, bottom, left) in enumerate(face_locations):
            # Draw a box around the face
            face_locations_all = {
                "frame": frame_id,
                "face": j,
                "bb": (left, top, right, bottom),
                "landmarks": face_locations[j],
            }

            face = Image.fromarray(rgb_frame)
            w, h = right - left, bottom - top
            left, right, bottom, top = (
                left - w * 0.1,
                right + w * 0.1,
                bottom + h * 0.1,
                top - h * 0.1,
            )
            face = face.crop([left, top, right, bottom])

            try:
                unknown_encoding = face_recognition.face_encodings(np.array(face))
                #     cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                # )
                if not len(unknown_encoding):
                    continue
                unknown_encoding = unknown_encoding[0]
                results = face_recognition.compare_faces(encodings, unknown_encoding)
                face.save(f"{dir_out}fr{frame_id}_face{j}.png")
                pd.to_pickle(
                    unknown_encoding, f"{dir_out}fr{frame_id}_face{j}-encoding.csv"
                )
                pd.DataFrame(results).astype(int).to_csv(
                    f"{dir_out}fr{frame_id}_face{j}-predictions.csv",
                    header=None,
                    index=False,
                )
                pd.DataFrame().from_dict(face_locations_all.items()).T.to_json(
                    f"{dir_out}fr{frame_id}_face{j}-meta.json"
                )

                print(results)
            except:
                print(f"{dir_out}fr{frame_id}_face{j}.png")
            finally:

                # cv2.imwrite(f"{dout}fr{frame_id}_face{j}.png", face)
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                frame_id += 1


def main():
    f_urls = "../../data/family_members.csv"

    if fetch_videos:
        fetch_videos(f_urls)

    f_video = "../../data/videos-fiw/Ben_Affleck.mp4"
    f_encodings = f"{str(Path(f_video).parent)}/mid-encodings.pkl"
    d_mid = Path(f_video).parent

    encodings = encode_mids(f_video=f_video, f_encodings=f_encodings)
    encodings = list(encodings.values())
    dout = f"{str(Path(f_video).parent)}/frames/"
    Path(dout).mkdir(exist_ok=True)

    process_subject(encodings, f_video, dout)


# import pickle
# with open(dout + "flandmarks.pkl", 'wb') as f:
#     pickle.dump(f)
# pd.to_pickle(face_locations_all, )


# virtualenv --no-site-packages ~/.virtualenvs/RingNet
# source ~/.virtualenvs/RingNet/bin/activate
# pip install --upgrade pip==19.1.1
if __name__ == "__main__":
    main()
