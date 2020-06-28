import glob
import pickle
from pathlib import Path
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

PACKAGE_PARENT = "../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
print(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import sys

sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from mtcnn import MTCNN
from tqdm import tqdm

dir_data = Path("/home/jrobby/VIDs-aligned-tp-frames/")
dir_videos = dir_data.glob("F????/MID*/v?????/track-*")
detector = MTCNN()

for i, dir_vid in tqdm(enumerate(dir_videos)):
    imfiles = list(Path(dir_vid).glob("*.jpg"))
    images = {
        str(imfile).replace(str(dir_data), ""): cv2.cvtColor(
            cv2.imread(str(imfile)), cv2.COLOR_BGR2RGB
        )
        for imfile in imfiles
    }
    imdetections = {}
    for ref, im in images.items():
        detections = detector.detect_faces(im)
        for j, detection in enumerate(detections):
            fout = dir_vid.joinpath(Path(ref).stem + "-" + str(j).zfill(2)).with_suffix(
                ".pkl"
            )
            with open(fout, "wb") as f:
                pickle.dump(detection, f)

        imdetections[ref] = detections

    with open(dir_vid.joinpath("detections.pkl"), "wb") as f:
        print("saved", len(imdetections))
        pickle.dump(imdetections, f)
