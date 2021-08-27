#
# Script to fuse features per member per family (i.e., for each FID.MID, average all encodings across feature dim).
#
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tools.features import l2_norm

dir_features = str(Path("./").home() / "datasets/rfiw2021/rfiw2021-data/FIDs-features/")
dir_out = ""
ext = "npy"  # ["pkl', 'npy']
# assume input/output directories are the same if no output is specified
dir_out = dir_out if len(dir_out) == 0 else dir_features
path_features = Path(dir_features)
dir_contents = list(path_features.glob("F????"))
normalize_features = True
do_pickle2numpy = False
# convert pkl files to npy (not required, just done if preferred).
# Average fuse all embeddings for each MID
for fid in tqdm(dir_contents):
    # for each FID
    print(f"FID: {fid}")
    for mid in fid.glob("MID*"):
        # for each member
        print(f"Fusing: {mid}")
        if not mid.is_dir():
            continue
        fout = mid / "avg_encoding.npy"
        features = []

        for face_feat in mid.glob(f"*face*.{ext}"):
            # for each face
            feature = None
            if ext == "pkl":
                try:
                    with open(str(face_feat), "rb") as fin:
                        feature = pickle.load(fin)
                        feature = np.array(feature)
                    if do_pickle2numpy:
                        np.save(str(face_feat).replace(".pkl", ".npy"), feature)
                except:
                    print(
                        f"WARNING: Exception thrown converting pickle to npy. {face_feat}"
                    )
            elif ext == "npy":
                feature = np.load(str(face_feat))
            else:
                # TODO : have as assert outside for loop (i.e., when value is set), but quick solution for now
                print(f"extension {ext} is unrecognizable. Options: [pkl, npy]")
                exit(0)
            if feature:
                features.append(feature)

        if features and normalize_features:
            # if features exist and normalize flag is set True
            features = np.mean(features, axis=0)
            features = l2_norm(features[None, ...])[0]

        if features.shape[0] == 512:
            print(f"Saving: {fout}")
            np.save(fout, features)
