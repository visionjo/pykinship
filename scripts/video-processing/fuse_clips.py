import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tools.features import l2_norm

path_feature_root = Path("/Volumes/MyWorld/FIW-MM/data/features/visual/video/arcface")
path_out1 = Path(
    "/Volumes/MyWorld/FIW-MM/data/features/visual/video/fused/arcface-fused-videos-avg-pooled"
)
# path_out2 = Path("/Volumes/MyWorld/FIW-MM/data/features/visual/video/fused/arcface-fused-video-clips-avg-pooled")
path_out2 = Path(
    "/Volumes/MyWorld/FIW-MM/data/features/visual/video/fused/arcface-fused-clips-avg-pooled"
)

path_out1.mkdir(exist_ok=True)
# path_out2.mkdir(exist_ok=True)
path_out2.mkdir(exist_ok=True)

paths_fids = list(path_feature_root.glob("F????"))
fids = [f.name for f in paths_fids]

for path_fid in tqdm(paths_fids):
    path_mids = list(path_fid.glob("MID*"))

    for path_mid in path_mids:
        clips = os.listdir(path_mid)
        clips = [c for c in clips if Path(path_mid / c).is_dir()]

        clip_ids = np.unique([c.split("_")[0] for c in clips])

        path_out_vid = Path(
            str(path_mid).replace(str(path_feature_root), str(path_out1))
        )

        for clip_id in clip_ids:
            fout = path_out_vid.joinpath(clip_id).with_suffix(".npy")
            if fout.is_file():
                continue

            Path(fout).parent.mkdir(parents=True, exist_ok=True)

            cur_clips = [c for c in clips if clip_id in c]

            paths_feats = [path_mid / c for c in cur_clips]

            mean_features = []
            # allfeatures = []
            for path_feats in paths_feats:
                files_features = list(path_feats.glob("fr*.npy"))
                features = np.array(
                    [np.load(f, allow_pickle=True) for f in files_features]
                )
                mean_feature = np.mean(features, axis=0).reshape((-1, 1))
                fout = Path(
                    str(path_feats).replace(str(path_feature_root), str(path_out2))
                ).with_suffix(".npy")
                Path(fout).parent.mkdir(parents=True, exist_ok=True)
                np.save(fout, l2_norm(mean_feature))
                mean_features.append(mean_feature)
                # allfeatures.append(features)

            mean_of_means = np.mean(np.array(mean_features), axis=0)
            np.save(fout, l2_norm(mean_of_means))

            # video_mean = np.mean(np.concatenate(allfeatures), axis=0).reshape((-1, 1))
            # fout = str(fout).replace(str(path_out2), str(path_out1))
            # Path(fout).parent.mkdir(parents=True, exist_ok=True)
            # np.save(fout, l2_norm(video_mean))
