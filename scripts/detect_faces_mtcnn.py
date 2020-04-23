import glob
import pickle

import cv2
from mtcnn import MTCNN
from tqdm import tqdm

dir_data = "../data/fiw-videos/new-processed/"
""
dir_videos = glob.glob(f"{dir_data}F????/v?????/")
detector = MTCNN()
for i, dir_vid in tqdm(enumerate(dir_videos)):
    imfiles = glob.glob(f"{dir_vid}scenes/images/*.jpg")
    images = {
        imfile.replace(dir_data, ""): cv2.cvtColor(
            cv2.imread(imfile), cv2.COLOR_BGR2RGB
        )
        for imfile in imfiles
    }
    dir_out = "/".join(imfiles[0].split("/")[:-3]) + "/"
    imdetections = {}
    for ref, im in images.items():
        detections = detector.detect_faces(im)
        imdetections[ref] = detections

    with open(f"{dir_out}mtcnn_detections.pkl", "wb") as f:
        pickle.dump(imdetections, f)
