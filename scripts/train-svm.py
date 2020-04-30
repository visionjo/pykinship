from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm

np.random.seed(1222)
path_roots = Path("../data/fiw-mm/FIDs-MM/").resolve()

path_features = (path_roots / "../features/image/arcface").resolve()

fids = [p.name for p in path_features.glob("F????")]

ids_probe = np.random.randint(0, len(fids) - 1)
probe_fid = fids[ids_probe]

fids_gallery = fids
del fids_gallery[ids_probe]

mids_probe_fid = list(path_features.joinpath(probe_fid).glob("MID*"))

ids_mid = np.random.randint(0, len(mids_probe_fid) - 1)

probe_mid = mids_probe_fid[ids_mid]

probe_features = pd.read_pickle(probe_mid.joinpath("encodings.pkl"))

gallery_features = np.array(
    [
        np.array(
            list(pd.read_pickle((path_features / f).joinpath("encodings.pkl")).values())
        )[0]
        for p in fids_gallery
        for f in Path(path_features / p).glob("MID*")
    ]
)

model = svm.SVC(C=10, kernel="rbf")

p_encodings = np.array(list(probe_features.values()))

labels = np.concatenate(
    [np.ones((len(p_encodings))), np.zeros((len(gallery_features)))]
)

model.fit(np.concatenate([p_encodings, gallery_features]), labels)
