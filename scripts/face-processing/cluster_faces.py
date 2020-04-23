import glob

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# # ap.add_argument("-e", "--encodings", required=True,
# #                 help="path to serialized db of facial encodings")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
#                 help="# of parallel jobs to run (-1 will use all CPUs)")
# args = vars(ap.parse_args())
dir_features = "../data/fiw-videos/new-processed/"
feat_files = glob.glob(dir_features + "F0008/v00000/scenes/faces/*.pkl")
im_files = glob.glob(dir_features + "F0008/v00000/scenes/faces/*.png")
feat_files.sort()
im_files.sort()
encodings = [pd.read_pickle(f) for f in feat_files]
# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=8)  # , n_jobs=args["jobs"])
clt.fit(encodings)
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))
imf = np.array(feat_files)[clt.labels_ == 0]
# build_montages()
# imf = [cv2.imread(f.replace('-encoding'))]

imf = [f.replace("-encoding.pkl", "") for f in imf]

imf = [
    "-".join(f.split("-")[:-1]) + "-{:02d}.png".format(int(f.split("-")[-1]))
    for f in imf
]

images = [cv2.imread(f) for f in imf]
