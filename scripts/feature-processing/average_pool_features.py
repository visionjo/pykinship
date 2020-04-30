from pathlib import Path

import numpy as np
from tqdm import tqdm

path_tracks = Path("/Volumes/MyWorld/FIW-MM/clips-tp-faces")
for fid in tqdm(list(path_tracks.iterdir())):
    if ".zip" in str(fid):
        continue
    for mid in fid.iterdir():
        if not mid.is_dir():
            continue
        for vid in mid.iterdir():
            if vid.is_dir():
                for vdir in vid.iterdir():
                    if vdir.is_dir():
                        fout = vdir / "avg_encoding.npy"
                        if fout.is_file():
                            print("skip")
                            continue
                        paths_encodings = []
                        for f_feature in vdir.glob("v*/*.npy"):
                            if "encoding" in f_feature.name:
                                continue
                            print(f_feature)
                            paths_encodings.append(f_feature)
                        if paths_encodings:
                            encodings = np.array(
                                [np.load(str(f)) for f in paths_encodings]
                            )
                            encodings = np.mean(encodings, axis=0)
                            if encodings.shape[0] == 512:
                                np.save(fout, encodings)
