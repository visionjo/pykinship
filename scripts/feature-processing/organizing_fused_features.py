from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

do_voxceleb = True
if not do_voxceleb:
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

else:
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
