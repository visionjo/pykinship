import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from PIL import Image
from moviepy.editor import VideoFileClip
from sklearn.utils import shuffle
from tqdm import tqdm

from src.align.align_trans import get_reference_facial_points, warp_and_crop_face
from src.align.detector import detect_faces
from src.align.get_nets import PNet, ONet, RNet

FPS = 25.0
DIGITS = 3
MIN_FACES = 10
MAX_FACES = 50

IMSIZE = (112, 112)
CROP_SIZE = 112


def get_cropped_faces(bboxes, frame_ids):
    frame_ids -= np.min(
        frame_ids
    )  # shift frame count to start at zero (i.e., start of the clip)

    nfaces = len(bboxes)
    # sample subset of faces to do comparison with
    nsamples = np.max([MIN_FACES, int(nfaces * 0.2)]) if MIN_FACES < nfaces else nfaces
    nsamples = np.min([nsamples, MAX_FACES])

    ids = np.random.randint(0, nfaces, nsamples)

    faces = []
    for bb, fid in zip(bboxes, frame_ids):
        # crop faces
        frame = video.get_frame(fid)
        im_pil = Image.fromarray(frame)

        hper, wper = bb[2:] * 0.1
        bb[0] = bb[0] - hper / 2
        bb[1] = bb[1] - wper / 2
        bb[2] = bb[2] + hper
        bb[3] = bb[3] + wper

        imcropped = im_pil.crop(bb)
        faces.append(imcropped)
    return list(np.array(faces)[ids])


def prepare_data(path_clips, f_lut):
    df = pd.DataFrame(path_clips, columns=["path"])

    df["vid"] = df.path.apply(lambda x: x.name)

    df_lut = pd.read_csv(f_lut)
    df["fid"], df["mid"] = None, None
    for vid in df_lut.vid.unique():
        # for each video
        ids = df.vid == vid
        if not ids.sum():
            # skip if no aligned tracks detected in video
            continue
        df_cur = df_lut.loc[df_lut.vid == vid]
        df.loc[ids, "fid"] = df_cur.fid.values[0]
        df.loc[ids, "mid"] = df_cur["mid"].astype(str).values[0]

    df["fid_mid"] = df["fid"] + "/MID" + df["mid"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument(
        "-source_root",
        "--source_root",
        help="specify your source dir",
        default=Path("/Volumes/MyWorld/FIW-MM/VIDs-aligned"),
        type=Path,
    )
    parser.add_argument(
        "-track_root",
        "--track_root",
        help="specify your track dir",
        default=Path("../../tracks"),
        type=Path,
    )
    parser.add_argument(
        "-data_root",
        "--data_root",
        help="specify your source dir",
        default=Path("../../data/fiw-mm/data"),
        type=Path,
    )

    parser.add_argument(
        "-w",
        "--wildcard",
        help="wild-card to append directory",
        default="F????/v?????/scenes/cropped/*.npy",
        type=str,
    )

    parser.add_argument(
        "-dest_root",
        "--dest_root",
        help="specify your destination dir",
        default=Path("/Volumes/MyWorld/FIW-MM/VIDs-aligned-faces"),
        type=Path,
    )
    parser.add_argument(
        "-l",
        "--master_list",
        help="LUT for mid->vid",
        default=Path("lists/fiw-videos-master.csv"),
        type=Path,
    )

    args = parser.parse_args()

    path_vids = args.source_root  # specify your source dir
    path_tracks = args.track_root  # specify your track dir
    path_out = args.dest_root  # specify your destination dir
    path_out.mkdir(exist_ok=True, parents=True)
    path_data = args.data_root

    path_encodings = path_data / "features/image/arcface/"

    path_clips = list(path_vids.rglob("*.mp4"))
    path_clips.sort()
    print(len(path_clips))

    dir_clips = list(set([p.parent for p in path_clips]))
    dir_clips.sort()

    scale = CROP_SIZE / 112.0
    reference = get_reference_facial_points(default_square=True) * scale

    pnet = PNet()
    rnet = RNet()
    onet = ONet()

    df_meta = prepare_data(dir_clips, path_data / args.master_list)

    umids = list(df_meta["fid_mid"].unique())
    umids.sort(reverse=False)
    # umids = umids[250:]
    for mid in tqdm(umids):
        # for each subject with video data
        dir_mid = path_encodings / mid  # directory with MID embeddings

        obin = path_out / mid
        obin.mkdir(parents=True, exist_ok=True)
        try:
            # copy mid embeddings
            # encodings = pd.read_pickle(dir_mid.joinpath("encodings.pkl"))
            copyfile(dir_mid.joinpath("encodings.pkl"), obin.joinpath("encodings.pkl"))
        except FileNotFoundError as e:
            print(e)
            print("No features found for {}. Skipping...".format(mid))
        # arr_encodings = np.array(list(encodings.values()))  # from dict to array, and order does not matter (same MID)

        # filter out videos that were collected for respective MID
        df_cur = df_meta.loc[df_meta["fid_mid"] == mid]
        path_dclips = df_cur.path.values

        for path_dclip in path_dclips:
            path_clips = list(path_dclip.glob("*.mp4"))  # get all the clips for

            path_track = path_tracks / path_dclip.with_suffix(".pkl").name
            tracks = pd.read_pickle(path_track)

            for track, path_clip in zip(tracks, path_clips):

                track = track["track"]

                oobin = (
                    path_out.joinpath(mid) / path_clip.parent.name / path_clip.name
                ).with_suffix("")
                try:
                    oobin.mkdir(parents=True)
                except Exception as e:
                    print(f"skipping {oobin}", e)
                    continue
                video = VideoFileClip(str(path_clip))
                video.set_fps(FPS)

                face_set = get_cropped_faces(track["bbox"], track["frame"])
                face_set = shuffle(face_set)
                new_detections = []
                faces_to_keep = []
                true_counter = 1
                for i, face in enumerate(face_set):
                    if true_counter > MAX_FACES:
                        break
                    try:
                        detections = detect_faces(face, pnet=pnet, rnet=rnet, onet=onet)
                    except Exception:
                        continue
                    if len(detections[0]):
                        faces_to_keep.append(face)
                        new_detections.append(detections)
                        true_counter += 1

                processed_faces = []
                counter = 0
                for detection, face in zip(new_detections, faces_to_keep):
                    id_max = 0
                    # for r in range(len(detection[0])):
                    if len(detection[0]) > 1:
                        id_max = np.argmax([l[-1] for l in detection[0]])
                    # if detection[0][r][-1] < 0.9:
                    #     continue
                    points = detection[1][id_max]
                    facial5points = [[points[j], points[j + 5]] for j in range(5)]

                    warped_face = warp_and_crop_face(
                        np.array(face),
                        facial5points,
                        reference,
                        crop_size=(CROP_SIZE, CROP_SIZE),
                    )
                    # processed_faces.append(warped_face)
                    fout = oobin / f"face-{str(counter).zfill(DIGITS)}.jpg"
                    counter += 1
                    Image.fromarray(warped_face).save(fout)
                true_counter += 1
