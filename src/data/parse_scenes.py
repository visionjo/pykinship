from glob import glob

import shutil
import sys

# if not "../../src" in sys.path:
#     sys.path.append("../..")
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.video import meta_face_location_to_bb, bb_intersection_over_union


def make_dirs(li_dirs):
    [Path(d).mkdir(exist_ok=True, parents=True) for d in li_dirs]


dir_fids = "/home/jrobby/master-version/fiwdb/FIDs/"  # '/Users/jrobby/data/FIDs/'
# df = read_family_member_list(f_urls)

dir_root = "/Volumes/MySpace/kinship/"

imimages = glob(dir_root + "*/scenes/faces/*.png")
imimages.sort()

dirs_scenes = [str(d) for d in Path(dir_root).glob("*") if Path(d).is_dir()]

dir_faces = [
    str(d) + "/scenes/faces/"
    for d in dirs_scenes
    if Path(str(d) + "/scenes/faces/").is_dir()
]
dir_scenes_parsed = [f.replace("/scenes/faces", "/scenes_parsed/") for f in dir_faces]
make_dirs(dir_scenes_parsed)


def copy_matched_data(files_in, files_out):
    # print(files_out)
    [
        shutil.copy(fin.replace("-meta.json", ".png"), fout)
        for fin, fout in zip(files_in, files_out)
    ]
    fout = "-".join(files_out[0].split("-")[:-1]) + "-bb.csv"
    pd.DataFrame(meta).to_csv(fout, header=False, index=False)


for dir_face in dir_faces:
    files_bb = np.array(glob(dir_face + "*-meta.json"))
    files_bb.sort()

    shot_ids = np.unique([f.split("-")[-4] for f in files_bb])
    shot_ids.sort()
    arr_lut = np.array([f.split("-")[-4] for f in files_bb])
    print(arr_lut)
    for shot_id in shot_ids:
        ids = shot_id == arr_lut

        f_cur_shot = np.array(list(files_bb[ids]))
        meta = {f.split("/")[-1]: meta_face_location_to_bb(f) for f in f_cur_shot}
        meta = {k: v for k, v in meta.items() if v}

        predictions = {
            f.split("/")[-1]: np.loadtxt(
                f.replace("-meta.json", "-predictions.csv"), delimiter=","
            )
            .astype(bool)
            .flatten()
            for f in f_cur_shot
        }

        npredictions = len(predictions)
        print(len(predictions), predictions)

        if not np.any(predictions):
            # case 1, no True predictions in shot
            continue

        print(f_cur_shot)
        fouts = [
            f.replace("-meta.json", ".png").replace("/scenes/faces", "/scenes_parsed/")
            for f in f_cur_shot
        ]
        if npredictions == 3:
            if np.sum([k for k in predictions.values()]) > 1:
                # case 2, same face in all of 3 thumbnails of shot
                copy_matched_data(f_cur_shot, fouts)
            continue
            # elif np.sum(predictions) == 2:
            #     # case 3, two of 3 shots true ID'd
            #     copy_matched_data(f_cur_shot, fouts)
            # continue
        df = pd.DataFrame(tuple(meta.items()))
        df2 = pd.DataFrame(tuple(predictions.items()))
        df.set_index(0, inplace=True)
        df["p"] = False
        df2.set_index(0, inplace=True)
        df.loc[df2.index, "p"] = df2[1]
        df.columns = ["bb", "p"]
        df.reset_index(inplace=True)
        df["shot"] = df.apply(lambda x: x[0].split("-")[1], axis=1)
        df["clip"] = df.apply(lambda x: x[0].split("-")[2], axis=1)
        df["face"] = df.apply(lambda x: x[0].split("-")[3], axis=1)

        # np.zeros((len(df), len(df)))
        # {k: v for k, v, p in zip(meta.items(), predictions.keys()) if p}
        thumbnail_ids = np.unique([f.split("-")[-3] for f in f_cur_shot])
        df["nn"] = None
        prev = None
        for thumbnail_id in thumbnail_ids:
            df_clip = df.loc[df["clip"] == thumbnail_id]
            if prev is not None:
                for k, row in df_clip.iterrows():
                    print(row)
                    ious = prev.apply(
                        lambda x: bb_intersection_over_union(x["bb"], row["bb"]), axis=1
                    )
                    ids_nn = ious.argmax()
                    if ious.iloc[ids_nn] > 0.6 and len(df_clip) > ids_nn:
                        df.loc[prev.iloc[[ids_nn]].index, "nn"] = df_clip.iloc[
                            [ids_nn]
                        ].index
            prev = df_clip.copy()

        df.to_csv(
            f_cur_shot[0]
            .replace("scenes/faces/", "/scenes_parsed/")
            .replace("-meta.json", "")[:-5]
            + ".csv"
        )
        if np.any(np.sum([k for k in predictions.values()], axis=1) == 3):
            continue
        if np.mean(np.array(predictions).mean()) > 0.9:
            fouts = [f for f in f_cur_shot]
            print(fouts)
            [
                shutil.copy(fin.replace("-meta.json", ".png"), fout)
                for fin, fout in zip(f_cur_shot, fouts)
            ]

            fout = "-".join(fouts[0].split("-")[:-1]) + "-bb.csv"
            pd.DataFrame(meta).to_csv(fout, header=False, index=False)
