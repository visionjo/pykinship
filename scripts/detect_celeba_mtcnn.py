import glob
import pickle
from pathlib import Path
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

dir_data = "../data/fiw-videos/new-processed/"
imfiles = [
    f for f in glob.glob(f"{dir_data}F????/MID*/faces/msceleb*") if Path(f).is_file()
]

images = {
    imfile.replace(dir_data, ""): cv2.cvtColor(cv2.imread(imfile), cv2.COLOR_BGR2RGB)
    for imfile in imfiles
}

detector = MTCNN()
for ref, im in tqdm(images.items()):
    imdetections = {}
    for ref, im in images.items():
        detections = detector.detect_faces(im)
        imdetections[ref] = detections

with open(f"{dir_data}mtcnn_celeba_detections.pkl", "wb") as f:
    pickle.dump(imdetections, f)
