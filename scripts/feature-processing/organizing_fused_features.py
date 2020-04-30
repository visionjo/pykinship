from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
import numpy as np

do_rsync = True
do_voxceleb = False
do_vid = True
if do_vid:
    dir_fids = Path("/Volumes/MyWorld/FIW-MM/clips-tp-faces/fused-features/")

    f_features = list(dir_fids.glob("F????/MID*/v*/*/avg_encoding.npy"))

    # Path(f_features[0]).symlink_to()

    dout = (
        Path.home()
        .joinpath("Dropbox")
        .joinpath("FIW_Video")
        .joinpath("data")
        .joinpath("features")
        .joinpath("video")
        .joinpath("avg-fused")
    )
    dout.mkdir(exist_ok=True, parents=True)

    for f_feature in tqdm(f_features):
        fout = Path(str(f_feature).replace(str(dir_fids), str(dout))).parent
        fout = fout.parent.parent / Path(fout.name).with_suffix(".npy")
        fout.parent.mkdir(parents=True, exist_ok=True)
        copyfile(f_feature, fout)

elif do_voxceleb:
    dir_fids = Path(
        "/Volumes/MyWorld/FIW-MM/features/visual/video/arcface-avg-fused/arcface-avg-fused/arcface/"
    )
    f_features = list(dir_fids.glob("F????/MID*/*/mean_encoding.npy"))
    dout = (
        Path.home()
        .joinpath("Dropbox")
        .joinpath("FIW_Video")
        .joinpath("data")
        .joinpath("features")
        .joinpath("video")
        .joinpath("avg-fused")
    )
    dout.mkdir(exist_ok=True, parents=True)

    for f_feature in tqdm(f_features):
        fout = Path(
            str(f_feature).replace(str(dir_fids), str(dout))
        ).parent.with_suffix(".npy")
        fout.parent.mkdir(parents=True, exist_ok=True)
        copyfile(f_feature, fout)

elif do_rsync:
    dir_fids = Path("/Volumes/MyWorld/FIW-MM/clips-tp-faces/fused-features/")

    f_features = dir_fids.glob("F????/MID*/v*/*/avg_encoding.npy")

    # Path(f_features[0]).symlink_to()

    dout = (
        Path.home()
        .joinpath("Dropbox")
        .joinpath("FIW_Video")
        .joinpath("data")
        .joinpath("FIDs-MM-features")
        .joinpath("visual")
        .joinpath("video")
        .joinpath("avg-fused")
    )
    dout.mkdir(exist_ok=True, parents=True)
    counter_new, counter_revised = 0, 0
    for f_feature in tqdm(f_features):
        fout = Path(str(f_feature).replace(str(dir_fids), str(dout))).parent
        fout = fout.parent.parent / Path(fout.name).with_suffix(".npy")
        if not fout.is_file():
            fout.parent.mkdir(parents=True, exist_ok=True)
            copyfile(f_feature, fout)
            counter_new += 0
            continue
        arr_in = np.load(f_feature)
        arr_saved = np.load(fout)

        if np.any(np.max(arr_in - arr_saved)):
            counter_new += 1
            copyfile(f_feature, fout)
