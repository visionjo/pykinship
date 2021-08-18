import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.utils import evaluate

dir_root = "/media/jrob/Seagate Backup Plus Drive/"
file_probe = f"{dir_root}rfiw2021/rfiw2021-data/track-III/test/data/lists/probe_lut.csv"
file_gallery = (
    f"{dir_root}rfiw2021/rfiw2021-data/track-III/test/data/lists/metadata_luts.csv"
)

dir_features = f"{dir_root}rfiw2021/rfiw2021-data/FIDs-features/"
path_features = Path(dir_features)

df_probes = pd.read_csv(file_probe)

df_probes["feat_path"] = df_probes.apply(
    lambda row: path_features / str(row["original"]), axis=1
)
df_probes["feat_path"] = df_probes.apply(
    lambda row: row["feat_path"] / "avg_encoding.npy", axis=1
)

fpaths = df_probes.feat_path.to_list()
feat_mat = []
labels = []
refs = []
for row in df_probes.iterrows():
    feat = np.load(str(row[1]["feat_path"]))
    feat_mat.append(feat)
    labels.append(row[1]["fid"])
    refs.append(row[1]["original"])
labels = np.array(labels)
feat_mat = np.array(feat_mat)

df_gallery = pd.read_csv(file_gallery)

df_gallery["feat_path"] = df_gallery.apply(
    lambda row: path_features / str(row["original"]).replace(".jpg", ".pkl"), axis=1
)
# df_gallery['feat_path'] = df_gallery.apply(lambda row : row['feat_path'] / 'avg_encoding.npy', axis = 1)

feat_mat_gallery = []
labels_gallery = []
refs_gallery = []
for row in df_gallery.iterrows():
    try:
        with open(str(row[1]["feat_path"]), "rb") as file:
            feat = pickle.load(file)
        # feat=np.load(str(row[1]['feat_path']))
        feat_mat_gallery.append(feat)
        labels_gallery.append(row[1]["fid"])
        refs_gallery.append(row[1]["original"])
    except:
        print(str(row[1]["feat_path"]))

labels_gallery = np.array(labels_gallery)
feat_mat_gallery = np.array(feat_mat_gallery)
## query-gallery
CMC = 0
ap = 0.0
all_scores = []
all_predicts = []
for i in range(labels.shape[0]):
    scores, predicts, (ap_tmp, CMC_tmp) = evaluate(
        feat_mat[i][None, ...], labels[i], feat_mat_gallery, labels_gallery
    )
    # all_scores.append(scores.squeeze())
    all_scores.append(scores)
    all_predicts.append(predicts)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp.numpy()
    ap += ap_tmp

CMC = CMC.astype(float)
CMC = CMC / labels.shape[0]  # average CMC
print("Rank@1:%f Rank@5:%f Rank@10:%f" % (CMC[0], CMC[4], CMC[9]))
print("Rank@10:%f Rank@20:%f Rank@50:%f" % (CMC[9], CMC[19], CMC[49]))
print("mAP:%f" % (ap / labels.shape[0]))

# save all_scores to npy
predict_result = {"score": np.asarray(all_scores), "predict": np.asarray(all_predicts)}
np.save("predict_result.npy", predict_result)

# CMC = CMC.numpy()
fig, ax = plt.subplots()
plt.plot(CMC)
ax.set(xscale="log")
plt.xlim(0, 1000)
plt.show()
fig.savefig("CMC_result.png")
