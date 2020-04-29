import argparse
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copyfile
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument(
        "-source_root",
        "--source_root",
        help="specify your source dir",
        default="/Volumes/MyWorld/FIW-MM/processed/",
        type=str,
    )
    parser.add_argument(
        "-data_root",
        "--data_root",
        help="specify your source dir",
        default="../../data/fiw-mm/",
        type=str,
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
        default="/Volumes/MyWorld/FIW-MM/processed-matches/",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--master_list",
        help="LUT for mid->vid",
        default="../../data/fiw-mm/lists/fiw-videos-master.csv",
        type=str,
    )

    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    path_out = Path(args.dest_root)  # specify your destination dir
    wildcard = args.wildcard
    path_out.mkdir(exist_ok=True, parents=True)

    path_data = Path(args.data_root).resolve()

    path_mids = path_data / "FIDs-MM/visual/image"
    path_detections = path_data / "interm/visual/video-frame-faces/"
    path_encodings = path_data / "features/image/arcface/"

    # cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    # os.chdir(source_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")
    # os.chdir(cwd)

    dir_features = list(
        set([Path(f).parent for f in glob.glob(f"{source_root}{wildcard}")])
    )
    dir_features.sort()

    df = pd.DataFrame(dir_features, columns=["path"])
    df["vid"] = df.path.apply(lambda x: x.parent.parent.name)
    df["fid"] = df.path.apply(lambda x: x.parent.parent.parent.name)

    df_lut = pd.read_csv(args.master_list)
    df_lut["fid_mid"] = df_lut["fid"] + "/MID" + df_lut["mid"].astype(str)
    print()

    umids = df_lut["fid_mid"].unique()
    umids.sort()

    for mid in tqdm(umids):
        dir_mid = path_encodings / mid

        obin = path_out / mid
        obin.mkdir(parents=True, exist_ok=True)
        try:
            encodings = pd.read_pickle(dir_mid.joinpath("encodings.pkl"))
        except FileNotFoundError as e:
            print(e)
            continue
        arr_encodings = np.array(list(encodings.values()))

        df_cur = df_lut.loc[df_lut["fid_mid"] == mid]

        path_features = np.array(dir_features)[df_cur.index.values.astype(int)]

        for epath in path_features:
            oobin = path_out / mid / epath.parent.parent.name
            oobin.mkdir(parents=True, exist_ok=True)

            for f_feature in epath.glob("*.npy"):
                face_encoding = np.load(f_feature)

                scores = cosine_similarity(
                    arr_encodings, face_encoding[..., np.newaxis].T
                )

                avg_score = np.mean(scores)

                if avg_score > 0.12:
                    copyfile(f_feature, oobin / f_feature.name)
                    copyfile(
                        f_feature.with_suffix(".jpg"),
                        oobin / Path(f_feature.name).with_suffix(".jpg"),
                    )
