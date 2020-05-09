import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

FPS = 25.0
DIGITS = 3
MIN_FACES = 10
MAX_FACES = 50

IMSIZE = (112, 112)
CROP_SIZE = 112


def prepare_data(path_clips):
    df = pd.DataFrame(path_clips, columns=["path"])

    df["mid"] = df.path.apply(lambda x: x.name)
    df["fid"] = df.path.apply(lambda x: x.parent.name)
    df["fid_mid"] = df["fid"] + "/MID" + df["mid"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument(
        "-source_root",
        "--source_root",
        help="specify your source dir",
        default=Path("/Volumes/MyWorld/FIW-MM/VIDs-aligned-faces"),
        type=Path,
    )
    parser.add_argument(
        "-track_root",
        "--track_root",
        help="specify your track dir",
        default=Path("/Volumes/MyWorld/FIW-MM/VIDs-aligned"),
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
        default=Path("/Volumes/MyWorld/FIW-MM/VIDs-aligned-tp"),
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

    path_source = args.source_root  # specify your source dir
    p_tracks = args.track_root  # specify your track dir
    path_out = args.dest_root  # specify your destination dir
    path_out.mkdir(exist_ok=True, parents=True)
    path_data = args.data_root

    path_encodings = list(path_source.glob("F????/MID*/encodings.pkl"))
    path_encodings.sort()

    for path_encoding in tqdm(path_encodings, total=len(path_encodings)):
        # for each subject with video data

        encodings = pd.read_pickle(path_encoding)
        arr_encodings = np.array(
            list(encodings.values())
        )  # from dict to array, and order does not matter (same MID)
        path_in = path_encoding.parent
        obin = Path(str(path_encoding.parent).replace(str(path_source), str(path_out)))
        obin.mkdir(exist_ok=True, parents=True)
        for path_track in path_in.glob("v?????/track*"):
            features_track = np.array([np.load(f) for f in path_track.glob("*.npy")])
            if not len(features_track):
                continue
            try:
                scores = cosine_similarity(arr_encodings, features_track)

                fused_score = np.median(scores, axis=0).mean()
            except Exception as e:
                print(e)
                pass
            if fused_score > 0.23:
                fin = (
                    p_tracks
                    / path_track.parent.name
                    / path_track.with_suffix(".mp4").name
                )

                fout = (
                    obin / path_track.parent.name / path_track.with_suffix(".mp4").name
                )
                fout.parent.mkdir(parents=True, exist_ok=True)
                copyfile(fin, fout)

        #
#         for f_feature in epath.glob("*.npy"):
#             face_encoding = np.load(f_feature)
#

#
#             avg_score = np.mean(scores)
#
#             if avg_score > 0.12:
#                 copyfile(f_feature, oobin / f_feature.name)
#                 copyfile(
#                     f_feature.with_suffix(".jpg"),
#                     oobin / Path(f_feature.name).with_suffix(".jpg"),
#                 )
#
# for path_clip in tqdm(path_clips, total=len(path_clips), unit="files"):
#     clip = VideoClip.VideoClip(path_clip)
#     ntracks = len(clip)
#     print(f"There are {ntracks} in {path_clip.name}")
#
#     f_source = path_vids / path_clip.with_suffix(".mp4").name
#     d_out = path_out / path_clip.with_suffix("").name
#     d_out.mkdir(exist_ok=True)
#     for k, track in enumerate(clip):
#         print(f"{k}/{ntracks}")
#         track = track["track"]
#         start_frame, end_frame = track["frame"][0], track["frame"][-1]
#         start_time, end_time = start_frame / FPS, end_frame / FPS
#         # int to string and add leading zeros
#         fout = d_out / f"track-{str(k).zfill(DIGITS)}.mp4"
#         ffmpeg_extract_subclip(f_source, start_time, end_time, targetname=f"{fout}")
