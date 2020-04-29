import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.model_irse import IR_152

PACKAGE_PARENT = "../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)

sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# # import src.tools.io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Helper function for extracting features from pre-trained models
# Download model weights on google drive:
# https://drive.google.com/file/d/1pA6iZYQ2i8BaVNn4ngXaKeIf6v50tk1m/view?usp=sharing


cuda, Tensor = (
    (True, torch.cuda.FloatTensor)
    if torch.cuda.is_available()
    else (False, torch.FloatTensor)
)


# Tensor = torch.FloatCuTensor
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode_faces(din, dout, ext=".jpg", tta=True):
    # src.tools.io.mkdir(exist_ok=True)

    imref = []
    arr_ccrop, arr_flip = None, None
    for imfile in din.glob(f"*{ext}"):
        image_name = str(imfile).replace(".jpg", ".npy")
        if Path(image_name).is_file():
            continue
        print("Processing\t{}".format(imfile))

        # load image
        img = cv2.imread(str(imfile))
        # os.remove(imfile)
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
        ccropped = ccropped.type(Tensor)
        if arr_ccrop is None:
            arr_ccrop = ccropped
        else:
            arr_ccrop = torch.cat((ccropped, arr_ccrop), 0)
            arr_ccrop = arr_ccrop.squeeze()

        flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
        flipped = np.reshape(flipped, [1, 3, 112, 112])
        flipped = np.array(flipped, dtype=np.float32)
        flipped = (flipped - 127.5) / 128.0
        flipped = torch.from_numpy(flipped)

        flipped = flipped.type(Tensor)

        if arr_flip is None:
            arr_flip = flipped
        else:
            arr_flip = torch.cat((flipped, arr_flip), 0)
            arr_flip = arr_flip.squeeze()
        imref.insert(0, imfile)
        # encodings = {}
        # if len(imref):
        #     src.tools.io.mkdir(exist_ok=True)
        if len(imref) == 256:
            with torch.no_grad():
                print(len(arr_ccrop))
                # while len(arr_ccrop) > 256:
                if tta:
                    # emb_batch = (
                    #         model(arr_ccrop[:256, ...]).cpu() + model(arr_flip[:256, ...]).cpu()
                    # )
                    emb_batch = model(arr_ccrop).cpu() + model(arr_flip).cpu()
                    features = l2_norm(emb_batch)
                else:
                    features = l2_norm(model(arr_ccrop)).cpu()
                    # features = l2_norm(model(arr_ccrop[:256, ...])).cpu()

                for feature, imfile in zip(features, imref):
                    # if not Path(image_name).is_file():
                    # continue
                    feature_arr = feature.data.cpu().numpy()
                    np.save(image_name, feature_arr)
                # features = np.load("features.npy")

                # ref = str(imfile).replace(source_root, "")
                # encodings[ref] = feature_arr
                # arr_ccrop = arr_ccrop[256:, ...]
                # arr_flip = arr_flip[256:, ...]
                del imref, features, feature_arr, emb_batch
                imref = []
                arr_ccrop, arr_flip = None, None
                # else:
    try:
        if len(arr_flip):
            with torch.no_grad():
                if tta:
                    emb_batch = model(arr_ccrop).cpu() + model(arr_flip).cpu()
                    features = l2_norm(emb_batch)
                else:
                    features = l2_norm(model(arr_ccrop)).cpu()

                for feature, imfile in zip(features, imref):
                    image_name = str(imfile).replace(".jpg", ".npy")

                    # if not Path(image_name).is_file():
                    # continue
                    feature_arr = feature.data.cpu().numpy()
                    np.save(image_name, feature_arr)
                    del feature_arr
                    # features = np.load("features.npy")

                    ref = str(imfile).replace(source_root, "")
                    print(ref)
                    # encodings[ref] = feature_arr
    except Exception as e:
        print("no remainder", e)

    return None


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
        default="/home/jrobby/clips-faces/",
        # default="/Users/jrobby/GitHub/pykinship/data/fiw-videos/FIDs-MM/",
        type=str,
    )
    parser.add_argument(
        "-dest_root",
        "--dest_root",
        help="specify your destination dir",
        # default="/Users/jrobby/GitHub/pykinship/data/fiw-videos/FIDs-MM-features/",
        default="/home/jrobby/clips-faces/",
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
    parser.add_argument(
        "-o", "--overwrite", help="overwrite if file exists", default=False, type=bool,
    )

    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root  # specify your destination dir
    overwrite = args.overwrite
    Path(dest_root).mkdir(exist_ok=True, parents=True)
    imsize = args.crop_size
    # model_root = "../model_ir_se50.pth"
    model_root = args.model_path
    print("Backbone Model Root:", model_root)

    # cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    # os.chdir(source_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")
    # os.chdir(cwd)

    dir_videos = list(
        set([Path(f).parent for f in glob.glob(f"{source_root}F????/v*/*/*.jpg")])
    )
    dir_videos.sort()
    # dir_videos = dir_videos[int(len(dir_videos) / 2):]
    # dir_videos = list(reversed(dir_videos))

    model = IR_152(imsize)
    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_root))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))

    # model.to(device)

    # extract features
    model.eval()  # set to evaluation mode

    # for subfolder in tqdm(os.listdir(source_root)):
    for subfolder in tqdm(dir_videos):
        # try:
        # path_out = Path(subfolder).joinpath("encodings")
        path_out = Path(subfolder)
        # path_out = Path(str(subfolder).replace('video-frame-faces/', 'video-frame-features/'))
        Path(path_out).mkdir(parents=True, exist_ok=True)
        path_in = Path(subfolder)
        # print(path_in, path_out)
        # if path_out.joinpath("encodings.pkl").is_file() and not overwrite:
        #     continue
        encodings = encode_faces(path_in, path_out)

        # with open(str(path_out.joinpath("encodings.pkl")), "wb") as f:
        #     pickle.dump(encodings, f)
