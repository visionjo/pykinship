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
        name, url, cmd="youtube-dl --continue --merge-output-format mp4 --all-subs -o {} {}", dir_out=None
):
    """
    Fetch video from youtube
    :param dir_out: directory to save to - if directory does not exist, then sets to save in current directory,
    :param name: identifier to name video with
    :param url: youtube URL to download as MP4
    :param cmd: command line call (Note, str.format() assumes 2 placeholders
    :return: True if successfully downloaded; else, return False.
    """
    if not Path(dir_out).is_dir():
        dir_out = ''

    try:

        os.system(cmd.format(dir_out + name, url))
        return True
    finally:
        return False


def encode_face_files(imfiles):
    images = OrderedDict({impath: cv2.imread(impath)[:, :, ::-1] for impath in imfiles})

    encodings = {}
    for impath, image in images.items():
        try:
            encodings[impath] = face_recognition.face_encodings(image)[0]
        except:
            print(f'Error encoding {impath}')
    return encodings


def read_family_member_list(f_csv):
    df = pd.read_csv(f_csv)
    # df['last'] = df["surname"].apply(lambda x: x.split('.')[0])
    df["ref"] = df["firstname"] + "_" + df["surname"]
    df = df.loc[df.video.notna()]
    df.reset_index(inplace=True)
    del df['index']
    return df


def fetch_videos(df, dir_out=None):
    df.apply(lambda x: wget_video(x["ref"], x["video"], dir_out=dir_out), axis=1)


def encode_mids(d_mid, f_encodings=None, save_pickle=True):
    if f_encodings and Path(f_encodings).is_file():
        encodings = pd.read_pickle(f_encodings)
    else:
        impaths = glob.glob(f"{d_mid}/*.jpg")
        encodings = encode_face_files(impaths)
        if save_pickle:
            f_encodings = f"{d_mid}/encodings.pkl"
            pd.to_pickle(encodings, f_encodings)
    return encodings


def crop_detection(face, locations):
    w, h = locations['bb'][2] - locations['bb'][0], locations['bb'][3] - locations['bb'][1]
    left, right, bottom, top = (
        locations['bb'][0] - w * 0.1,
        locations['bb'][2] + w * 0.1,
        locations['bb'][3] + h * 0.1,
        locations['bb'][1] - h * 0.1,
    )
    return face.crop([left, top, right, bottom])


def get_video_metadata(f_video):
    cap = cv2.VideoCapture(f_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f'{f_video} meta:\n=====')
    print(f'fps = {fps}')
    print(f'number of frames = {frame_count}')
    print(f'duration (S) = {duration}')
    print(f'duration (M:S) = {int(duration / 60)}:{duration % 60}')
    cap.release()


def process_subject(encodings, f_video, dir_out):
    get_video_metadata(f_video)
    video_capture = cv2.VideoCapture(f_video)
    # Check if camera opened successfully 
    if not video_capture.isOpened():
        print("Error opening video  file")
        return

    frame_id = 0
    # Read until video is completed 
    while video_capture.isOpened():
        print(Path(f"fr{frame_id}_face{0}.png"))

        print(dir_out)
        # Capture frame-by-frame 
        ret, frame = video_capture.read()
        if ret:
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]
            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

            # Initialize variables
            # face_locations_all = []
            # frame_id = 0
            # Display the results
            for j, (top, right, bottom, left) in enumerate(face_locations):
                # Draw a box around the face
                face_locations_dict = {
                    "frame": frame_id,
                    "face": j,
                    "bb": (left, top, right, bottom),
                    "landmarks": face_locations[j],
                }

                face_image = crop_detection(Image.fromarray(rgb_frame), face_locations_dict)
                # try:
                unknown_encoding = face_recognition.face_encodings(np.array(face_image))
                #     cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                # )
                if not len(unknown_encoding):
                    continue
                unknown_encoding = unknown_encoding[0]
                results = face_recognition.compare_faces(encodings, unknown_encoding)
                face_image.save(f"{dir_out}faces/fr{frame_id}_face{j}.png")
                pd.to_pickle(
                    unknown_encoding, f"{dir_out}encodings/fr{frame_id}_face{j}-encoding.csv"
                )
                pd.DataFrame(results).astype(int).to_csv(
                    f"{dir_out}predictions/fr{frame_id}_face{j}-predictions.csv",
                    header=None,
                    index=False,
                )
                pd.DataFrame().from_dict(face_locations_dict.items()).T.to_json(
                    f"{dir_out}meta/fr{frame_id}_face{j}-meta.json"
                )

                print(results)
                # except:
                print(f"{dir_out}fr{frame_id}_face{j}.png")
                # finally:

                # cv2.imwrite(f"{dout}fr{frame_id}_face{j}.png", face)
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            frame_id += 1
        else:
            break

    # When everything done, release the video capture object
    video_capture.release()


def process_sample(df, dir_videos, dir_encodings):
    f_video = dir_videos + df['ref'] + '.mp4'
    if not Path(f_video).is_file():
        return
    print(f_video)
    dir_mid = f"{dir_encodings}{df['fid']}/MID{df['mid']}/"
    encodings = encode_mids(dir_mid)
    encodings = list(encodings.values())
    dout = f"{dir_videos}{df['ref']}/"
    Path(dout).mkdir(exist_ok=True)
    Path(dout).joinpath('encodings').mkdir(exist_ok=True)
    Path(dout).joinpath('faces').mkdir(exist_ok=True)
    Path(dout).joinpath('predictions').mkdir(exist_ok=True)
    Path(dout).joinpath('meta').mkdir(exist_ok=True)
    process_subject(encodings, f_video, dout)


def main():
    f_urls = "../../data/family_members.csv"
    f_video = "../../data/videos-fiw/Ben_Affleck.mp4"
    f_encodings = f"{str(Path(f_video).parent)}/mid-encodings.pkl"
    dir_out = "/home/jrobby/kinship/"
    dir_fids = '/home/jrobby/master-version/fiwdb/FIDs/'
    df = read_family_member_list(f_urls)
    fetch_videos = False
    if fetch_videos:
        fetch_videos(df, dir_out)

    # f_videos = Path(dir_videos).glob('*.mp4')
    # d_mid = Path(f_video).parent
    nsubjects = len(df)
    for i in range(nsubjects):
        process_sample(df.iloc[i], dir_out, dir_fids)
    # df.apply(lambda x: , axis=1)
    # process_se(df)


# import pickle
# with open(dout + "flandmarks.pkl", 'wb') as f:
#     pickle.dump(f)
# pd.to_pickle(face_locations_all, )


# virtualenv --no-site-packages ~/.virtualenvs/RingNet
# source ~/.virtualenvs/RingNet/bin/activate
# pip install --upgrade pip==19.1.1
if __name__ == "__main__":
    main()
