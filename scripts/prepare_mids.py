import glob
import numpy as np
import pandas as pd
import face_recognition
import cv2
import tqdm

dir_data = "/home/jrobby/kinship/processed/"
dir_data = "/Volumes/MyWorld/FIW_Video/data/processed/"

dir_mids = "/Users/jrobby/master-version/fiwdb/FIDs/"

f_meta = glob.glob(dir_data + "*/fiw.txt")

fiw_meta = [(f, np.loadtxt(f, dtype=str)) for f in f_meta]

fiw_meta = [(f[0], str(f[1]).replace(".", "/MID")) for f in fiw_meta]

for meta in tqdm.tqdm(fiw_meta):
    fout = meta[0].replace("fiw.txt", "") + "fiw-encodings.pkl"
    dmid = dir_mids + meta[1] + "/"
    imfiles = glob.glob(dmid + "*.jpg")
    images = [cv2.imread(impath)[:, :, ::-1] for impath in imfiles]

    features = []
    for im in images:
        try:
            features.append(face_recognition.face_encodings(im)[0])
        except Exception as e:
            print("ERROR {}".format(e.message))
    # features = encode_mids
    if features:
        pd.to_pickle(features, fout)
