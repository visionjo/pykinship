# import libraries
import cv2
import os
import face_recognition
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import glob
from collections import OrderedDict
import tqdm


def wget_video(name, url, cmd="youtube-dl --continue --merge-output-format mp4 --all-subs -o {} {}"):
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

f_video = "../../data/videos-fiw/Ben_Affleck.mp4"
video_capture = cv2.VideoCapture(f_video)
impaths = glob.glob(f"{str(Path(f_video).parent)}/*.jpg")
images = OrderedDict({impath: cv2.imread(impath)[:, :, ::-1] for impath in impaths})
# face_locations = [face_recognition.face_locations(im) for im in images]

encodings = {
    path: face_recognition.face_encodings(images[path])[0] for path in images.keys()
}
pd.to_pickle(encodings, f"{str(Path(f_video).parent)}/mid-encodings.pkl")
encodings = list(encodings.values())
dout = f"{str(Path(f_video).parent)}/frames/"
Path(dout).mkdir(exist_ok=True)
# Initialize variables
face_locations_all = []
frame_id = 0
while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()
    if Path(f"{dout}fr{frame_id}_face{0}.png").exists():
        face_landmarks_list = face_recognition.face_landmarks(frame[:, :, ::-1])
        pd.DataFrame().from_dict(face_landmarks_list[0].items()).T.to_json(f"{dout}fr{frame_id}_face{0}-landmarks.json")
        frame_id += 1
        continue
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

    # Display the results
    for j, (top, right, bottom, left) in enumerate(face_locations):
        # Draw a box around the face
        face_locations_all = frame_id, j, (left, top, right, bottom), face_landmarks_list[j]

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
            face.save(f"{dout}fr{frame_id}_face{j}.png")
            pd.to_pickle(unknown_encoding, f"{dout}fr{frame_id}_face{j}-encoding.csv")
            pd.DataFrame(results).astype(int).to_csv(
                f"{dout}fr{frame_id}_face{j}-predictions.csv", header=None, index=False
            )
            pd.DataFrame().from_dict(face_locations_all[-1].items()).T.to_json(

                f"{dout}fr{frame_id}_face{j}-meta.json")
            pd.DataFrame().from_dict(face_landmarks_list.items()).T.to_json(
                f"{dout}fr{frame_id}_face{j}-landmarks.json")

            print(results)
        except:
            print(f"{dout}fr{frame_id}_face{j}.png")
        finally:

            # cv2.imwrite(f"{dout}fr{frame_id}_face{j}.png", face)
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            frame_id += 1

# import pickle
# with open(dout + "flandmarks.pkl", 'wb') as f:
#     pickle.dump(f)
# pd.to_pickle(face_locations_all, )


# download


# # df = pd.DataFrame()
# df.apply(lambda x: download_video(x['firstname']+'_'+x['surname'], x['video']), axis=1)
#
# virtualenv --no-site-packages ~/.virtualenvs/RingNet
# source ~/.virtualenvs/RingNet/bin/activate
# pip install --upgrade pip==19.1.1
