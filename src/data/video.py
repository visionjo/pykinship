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
from tqdm import tqdm


def wget_video(
    name,
    url,
    cmd="youtube-dl --continue --write-auto-sub --get-thumbnail --write-all-thumbnails --get-description --all-subs -o {} {}",
    dir_out=None,
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
        dir_out = ""

    try:
        if not glob.glob(dir_out + name + "/*.mp4") + glob.glob(
            dir_out + name + "/*.mkv"
        ):
            os.system(cmd.format(dir_out + name, url))
        return True
    finally:
        print(name)
        return False


def encode_face_files(imfiles):
    images = OrderedDict({impath: cv2.imread(impath)[:, :, ::-1] for impath in imfiles})

    encodings = {}
    for impath, image in images.items():
        try:
            encodings[impath] = face_recognition.face_encodings(image)[0]
        except Exception as e:
            print(f"Error encoding {impath} {e.message}")
    return encodings


def read_family_member_list(f_csv):
    df = pd.read_csv(f_csv)
    # df['last'] = df["surname"].apply(lambda x: x.split('.')[0])
    df["ref"] = df["firstname"] + "_" + df["surname"]
    df = df.loc[df.video.notna()]
    df.reset_index(inplace=True)
    del df["index"]
    return df


def fetch_videos(df, dir_out=None):
    df.apply(lambda x: wget_video(x["ref"], x["video"], dir_out=dir_out), axis=1)


def encode_mids(d_mid, f_encodings=None, save_pickle=False):
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
    w, h = (
        locations["bb"][2] - locations["bb"][0],
        locations["bb"][3] - locations["bb"][1],
    )
    left, right, bottom, top = (
        locations["bb"][0] - w * 0.1,
        locations["bb"][2] + w * 0.1,
        locations["bb"][3] + h * 0.1,
        locations["bb"][1] - h * 0.1,
    )
    return face.crop([left, top, right, bottom])


def get_video_metadata(f_video):
    cap = cv2.VideoCapture(f_video)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration"] = meta["frame_count"] / meta["fps"]
    cap.release()
    return meta


def print_video_metadata(f_video):
    cap = cv2.VideoCapture(f_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"{f_video} meta:\n=====")
    print(f"fps = {fps}")
    print(f"number of frames = {frame_count}")
    print(f"duration (S) = {duration}")
    print(f"duration (M:S) = {int(duration / 60)}:{duration % 60}")
    cap.release()


def process_subject(encodings, f_video, dir_out):
    print_video_metadata(f_video)
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

        if Path(f"{dir_out}faces/fr{frame_id}_face{0}.png").is_file():
            print("skipping")
            frame_id += 1
            continue
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

                face_image = crop_detection(
                    Image.fromarray(rgb_frame), face_locations_dict
                )
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
                    unknown_encoding,
                    f"{dir_out}encodings/fr{frame_id}_face{j}-encoding.csv",
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


def process_sample(df, dir_videos, encodings=None):
    f_video = dir_videos + df["ref"] + ".mp4"
    dout = f"{dir_videos}{df['ref']}/"
    if Path(dout).is_dir():
        return

    if not Path(f_video).is_file():
        return
    print(f_video)

    Path(dout).mkdir(exist_ok=True)
    Path(dout).joinpath("encodings").mkdir(exist_ok=True)
    Path(dout).joinpath("faces").mkdir(exist_ok=True)
    Path(dout).joinpath("predictions").mkdir(exist_ok=True)
    Path(dout).joinpath("meta").mkdir(exist_ok=True)
    process_subject(encodings, f_video, dout)
    os.remove(f_video)


def process_scenes(dir_in, dir_out, encodings=None):
    """ process first, middle, and last image of each scene to determine whether MID is present.
    :param
    """
    Path(dir_out).mkdir(exist_ok=True)
    imfiles = [f for f in glob.glob(dir_in + "*.jpg") if f.count("Scene-")]
    imfiles.sort()
    shot_ids = np.unique(np.array([f.split("-")[-2] for f in imfiles]))
    shots = np.unique(shot_ids)
    imfiles = np.array(imfiles)
    # imfiles = np.reshape(imfiles, (-1, 3))

    for i, shot_id in enumerate(shots):
        imstack = imfiles[np.where(shot_ids == shot_id)[0]]
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        for face_id, imfile in enumerate(imstack):
            suffix = imfile.split(".")[-2][-7:]
            frame = cv2.imread(imfile)
            rgb_frame = frame[:, :, ::-1]
            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

            for j, (top, right, bottom, left) in enumerate(face_locations):
                # Draw a box around the face
                face_locations_dict = {
                    "frame": shot_id,
                    "face": j,
                    "bb": (left, top, right, bottom),
                    "landmarks": face_locations[j],
                }

                face_image = crop_detection(
                    Image.fromarray(rgb_frame), face_locations_dict
                )
                # try:
                unknown_encoding = face_recognition.face_encodings(np.array(face_image))
                #     cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                # )
                if not len(unknown_encoding):
                    continue
                unknown_encoding = unknown_encoding[0]
                results = face_recognition.compare_faces(encodings, unknown_encoding)
                face_image.save(f"{dir_out}s{suffix}-{j}.png")
                pd.to_pickle(unknown_encoding, f"{dir_out}s{suffix}-{j}-encoding.csv")
                pd.DataFrame(results).astype(int).to_csv(
                    f"{dir_out}s{suffix}-{j}-predictions.csv", header=None, index=False,
                )
                pd.DataFrame().from_dict(face_locations_dict.items()).T.to_json(
                    f"{dir_out}s{suffix}-{j}-meta.json"
                )

                print(results)
                # except:
                print(f"{dir_out}s{suffix}-{j}.png")
                # finally:


def meta_face_location_to_bb(f_json):
    try:
        df_meta = pd.read_json(f_json)
        l, t, r, b = tuple(df_meta.iloc[1][2])
        bb = l, t, r, b
    except Exception as e:
        print(e.message)
        bb = None
    return bb


# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========
def bb_intersection_over_union(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # compute IoU by taking the intersection area and dividing it by the sum of BBs A + B areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_face_ids_from_image_list(li_impaths):
    ids = list(set([f.split(".")[-2].split("face")[-1] for f in li_impaths]))
    ids.sort()
    return ids


def calculate_iou_bb_neighboring_frames(din, fout, nframes):
    """
    :param din directory containing bb information (json)
    :param fout pickle file path to save array of iou values (K x F), where is K is number of frames and F is the max
                number of faces in a frame at a given instance.
    """
    f_metas = glob.glob(din + "*.json")
    ids = get_face_ids_from_image_list(f_metas)

    # i = "face" + ids[0]
    # cfaces = [f for f in f_metas if f.count(i)]
    # nframes = meta['frame_count']
    # cur_track = 0

    ids = [int(i) for i in ids]

    li_track_ids = []
    for n in ids:
        track_ids = -1 * np.ones((nframes,))
        box_prev = 0
        for k in range(nframes):
            fin = "{0}fr{1:06d}_face{2:02d}.json".format(din, k, n)

            if Path(fin).is_file():
                box_cur = meta_face_location_to_bb(fin)
                if box_prev:
                    iou = bb_intersection_over_union(box_cur, box_prev)

                    track_ids[k] = iou

                # if iou < .75:
                #     cur_track += 1
                # track_ids[k][1] = cur_track

                box_prev = box_cur
            # else:
            #     box_prev = -1
        li_track_ids.append(track_ids)
    pd.to_pickle(li_track_ids, fout)


# def main():
if __name__ == "__main__":
    f_urls = "../../data/family_members.csv"
    f_video = "../../data/videos-fiw/Ben_Affleck.mp4"
    f_encodings = f"{str(Path(f_video).parent)}/mid-encodings.pkl"
    dir_out = "/Volumes/MySpace/kinship/"
    dir_scenes = "/home/jrobby/kinship/"
    dir_fids = "/home/jrobby/master-version/fiwdb/FIDs/"  # '/Users/jrobby/data/FIDs/'
    df = read_family_member_list(f_urls)
    get_videos = False
    process_videos = False
    procese_shots = True
    if get_videos:
        fetch_videos(df, dir_out)

    # f_videos = Path(dir_videos).glob('*.mp4')
    # d_mid = Path(f_video).parent
    if process_videos:
        nsubjects = len(df)
        for i in range(nsubjects):
            dir_mid = f"{dir_fids}{df['fid']}/MID{df['mid']}/"
            encodings = encode_mids(dir_mid)
            encodings = list(encodings.values())
            process_sample(df.iloc[i], dir_out, encodings)
    # df.apply(lambda x: , axis=1)
    # process_se(df)
    if procese_shots:
        dir_data = "/home/jrobby/kinship/processed/"
        # dir_data = '/Volumes/MyWorld/FIW_Video/data/processed/'
        dirs_scenes = glob.glob(dir_data + "*/*/scenes/")
        dirs_scenes.sort()

        for dir_scene in tqdm(dirs_scenes):
            print(dir_scene)
            subject = dir_scene.split("/")[-4]
            dir_scene = dir_scene + "/"
            print(subject)
            dout = dir_scene + "/detections/"
            print(subject)
            encodings = pd.read_pickle(
                str(Path(dir_data).joinpath(subject)) + "/fiw-encodings.pkl"
            )

            process_scenes(dir_scene, dout, encodings)

    # if __name__ == "__main__":
    # main()

    # cmd = 'youtube-dl  --write-description --continue --write-sub --write-auto-sub --all-subs --write-all-thumbnails ' \
    # '-o {}  {}'
