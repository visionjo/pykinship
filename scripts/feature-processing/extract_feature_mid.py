"""
Helper function for extracting features from pre-trained models
"""
import glob

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.models.model_irse import IR_152

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tta = True


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face alignment")
    parser.add_argument(
        "-source_root",
        "--source_root",
        help="specify your source dir",
        default="../../data/fiw-videos/FIW-MM/",
        type=str,
    )
    parser.add_argument(
        "-dest_root",
        "--dest_root",
        help="specify your destination dir",
        default="../../data/fiw-videos/FIW-MM/",
        type=str,
    )
    parser.add_argument(
        "-model_path",
        "--model_path",
        help="specify path to model weights",
        default="../../models/Backbone_IR_152_checkpoint.pth",
        type=str,
    )

    parser.add_argument(
        "-im_size",
        "--crop_size",
        help="specify size of aligned faces, align and crop with padding",
        default=(112, 112),
        type=int,
    )
    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root  # specify your destination dir
    imsize = (
        args.crop_size
    )  # specify size of aligned faces, align and crop with padding
    # model_root = "../model_ir_se50.pth"
    model_root = args.model_path
    print("Backbone Model Root:", model_root)

    # cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    # os.chdir(source_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")
    # os.chdir(cwd)

    dir_mids = glob.glob(f"{source_root}*/F????.MID*")

    model = IR_152(imsize)
    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_root))
    else:
        model.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    model.to(device)

    # extract features
    model.eval()  # set to evaluation mode

    for dir_mid in tqdm(dir_mids):

        fid_mid = dir_mid.split("/")[-1].replace(".MID", "/MID")
        dout = Path(dest_root).joinpath(fid_mid)
        try:
            dout.mkdir()
        except Exception:
            continue

        arr_ccropped, arr_flipped = None, None
        imref = []
        for imfile in Path(dir_mid).glob("*.jpg"):

            print("Processing\t{}".format(imfile))
            # load image
            img = cv2.imread(imfile._str)

            # resize image to [128, 128]
            resized = cv2.resize(img, (128, 128))

            # center crop image
            a = int((128 - 112) / 2)  # x start
            b = int((128 - 112) / 2 + 112)  # x end
            c = int((128 - 112) / 2)  # y start
            d = int((128 - 112) / 2 + 112)  # y end
            ccropped = resized[a:b, c:d]  # center crop the image
            ccropped = ccropped[..., ::-1]  # BGR to RGB

            # flip image horizontally
            flipped = cv2.flip(ccropped, 1)

            # load numpy to tensor
            ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
            ccropped = np.reshape(ccropped, [1, 3, 112, 112])
            ccropped = np.array(ccropped, dtype=np.float32)
            ccropped = (ccropped - 127.5) / 128.0
            ccropped = torch.from_numpy(ccropped)

            if arr_ccropped is None:
                arr_ccropped = ccropped
            else:
                arr_ccropped = torch.cat([ccropped, arr_ccropped], 0)
                arr_ccropped = arr_ccropped.squeeze()

            flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
            flipped = np.reshape(flipped, [1, 3, 112, 112])
            flipped = np.array(flipped, dtype=np.float32)
            flipped = (flipped - 127.5) / 128.0
            flipped = torch.from_numpy(flipped)
            if arr_flipped is None:
                arr_flipped = flipped
            else:
                arr_flipped = torch.cat([flipped, arr_flipped], 0)
                arr_flipped = arr_flipped.squeeze()
            # images.append((ccropped, flipped))
            imref.insert(0, imfile)

        with torch.no_grad():
            if tta:
                emb_batch = (
                    model(arr_ccropped.to(device)).cpu()
                    + model(arr_flipped.to(device)).cpu()
                )
                encodings = l2_norm(emb_batch)
            else:
                encodings = l2_norm(model(arr_ccropped.to(device)).cpu())
        dic_encodings = {}
        for encoding, imfile in zip(encodings, imref):
            ref = str(imfile).replace(source_root, "")
            feat_name = imfile._str.split("/")[-1].replace(".jpg", ".npy")
            fout = dout.joinpath(feat_name)
            np.save(fout, encoding)

            # features = np.load("features.npy")
            dic_encodings[ref] = encoding
        pd.to_pickle(dic_encodings, str(Path(dout).joinpath("encodings.pkl")))
