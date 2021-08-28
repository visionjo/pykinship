import pickle
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import numpy as np
import pandas as pd
from evaluation.utils import evaluate
from numpy import array
from pandas import DataFrame
from tqdm import tqdm


def make_submission(
    arr: array, filepath: str = "predictions.csv", zipfile: str = "entry.zip"
) -> None:
    np.savetxt(filepath, arr, delimiter=",")
    # writing as zip file. SUBMIT ZIP ON CODALAB
    with ZipFile("my_python_files.zip", "w") as zipin:
        # writing each file one by one
        zipin.write(filepath)


def load_feature_file(path):
    """
    Load features stored as dictionary with fields 'feature_mat', 'labels', and 'refs'
    :param path:
    :return:
    """
    with open(path, "rb") as fin:
        features_in = pickle.load(fin)
    # mat_feat = features_in["feat_mat"]
    # arr_labels = features_in["labels"]
    # arr_refs = features_in["refs"]

    return features_in


def load_feature_files(df: DataFrame) -> Dict:
    # load all pickle files and dump to single pickle
    filepaths = df.feat_path.to_list()
    feat_mat = []
    for filepath in tqdm(filepaths):
        feat = np.load(str(filepath), allow_pickle=True)
        feat_mat.append(feat)

    return {
        "feat_mat": np.array(feat_mat),
        "labels": np.array(df["fid"].to_list()),
        "refs": df["original"].to_list(),
    }


def read_list_file(filename: str, list_type: str, dir_features: Path) -> DataFrame:
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    # read in probe list (table)
    df = pd.read_csv(filename)
    if list_type == "probes":
        # add column for file pointer to average encoding for MID
        df["feat_path"] = df.apply(
            lambda row: dir_features / str(row["original"]) / "avg_encoding.npy", axis=1
        )
    elif list_type == "gallery":
        df["feat_path"] = df.apply(
            lambda el: dir_features / str(el["original"]).replace(".jpg", ".pkl"),
            axis=1,
        )
    return df


experiment_name = "track-III-benchmark"
# set flags
overwrite = True
# set paths
path_dir_root = Path.home() / "datasets" / "rfiw2021"
path_dir_data = path_dir_root / "rfiw2021-data"
path_dir_test = path_dir_data / "track-III" / "test"
path_dir_features = path_dir_data / "FIDs-features"

path_file_probe = path_dir_test / "probe_list.csv"
path_file_gallery = path_dir_test / "gallery_list.csv"

path_file_probe_features = path_dir_test / "probe_features.pkl"
path_file_gallery_features = path_dir_test / "gallery_features.pkl"

path_dir_out = Path("./") / "results" / "task-III" / experiment_name
path_dir_out.mkdir(parents=True, exist_ok=True)
df_probes = read_list_file(str(path_file_probe), "probes", path_dir_features)

if not overwrite and path_file_probe_features.exists():
    # load if probe features are stored
    features_probes = load_feature_file(path_file_probe_features)
else:
    features_probes = load_feature_files(df_probes)
    with open(path_file_probe_features, "wb") as fout:
        pickle.dump(features_probes, fout)

df_gallery = read_list_file(str(path_file_gallery), "gallery", path_dir_features)

if not overwrite and path_file_gallery_features.exists():
    features_gallery = load_feature_file(path_file_gallery_features)
else:
    features_gallery = load_feature_files(df_gallery)
    with open(path_file_gallery_features, "wb") as fout:
        pickle.dump(features_gallery, fout)

## query-gallery
CMC = 0
ap = 0.0
all_scores = []

# all predictions are to be submitted (see below)
all_predicts = []
y_score = []
Y_test = []

for i in range(features_probes["labels"].shape[0]):
    scores, predicts, (ap_tmp, CMC_tmp) = evaluate(
        features_probes["feat_mat"][i][None, ...],
        features_probes["labels"][i],
        features_gallery["feat_mat"],
        features_gallery["labels"],
    )

    true_matches = df_gallery.loc[predicts, "fid"].values == df_probes["fid"][i]

    y_score.append(scores[0])
    Y_test.append(true_matches.astype(int))
    # all_scores.append(scores.squeeze())
    all_scores.append(scores)
    all_predicts.append(predicts)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

y_score = np.array(y_score)
Y_test = np.array(Y_test)
CMC = CMC.astype(float)
CMC = CMC / features_probes["labels"].shape[0]  # average CMC
print("Rank@1:%f Rank@5:%f Rank@10:%f" % (CMC[0], CMC[4], CMC[9]))
print("Rank@10:%f Rank@20:%f Rank@50:%f" % (CMC[9], CMC[19], CMC[49]))
print("mAP:%f" % (ap / features_probes["labels"].shape[0]))
all_scores = np.asarray(all_scores)
all_predicts = np.asarray(all_predicts)

make_submission(all_predicts)

# save all_scores to npy
predict_result = {
    "all_scores": all_scores,
    "Y_test": Y_test,
    "y_score": y_score,
    "all_predicts": all_predicts,
    "labels_gallery": features_gallery["labels"],
    "labels": features_probes["labels"],
    "CMC": CMC,
}
with open("predict_result.pkl", "wb") as file:
    pickle.dump(predict_result, file)
