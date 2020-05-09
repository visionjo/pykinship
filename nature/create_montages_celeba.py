from pathlib import Path

from tqdm import tqdm
import cv2
import face_recognition
from imutils import build_montages, opencv2matplotlib
from imutils.paths import list_images

import matplotlib.pyplot as plt


# compare embeddings
if __name__ == "__main__":
    do_mids = False
    encode_faces = False
    create_montages = True
    dir_data = "/Users/jrobby/data/nature/"

    dirs_fid = Path(dir_data).glob("F????")

    for dir_fid in tqdm(dirs_fid):
        dirs_mid = dir_fid.glob("MID*")
        if create_montages:
            paths = [l for l in list_images(dir_fid) if l.count("msceleb")]
            paths.sort()
            nmembers = len(set([Path(p).parent.parent.as_posix() for p in paths]))
            images = [cv2.imread(path) for path in paths]

            m = build_montages(images, (112, 108), (10, nmembers))
            im = opencv2matplotlib(m[0])
            # plt.axes(False)
            # plt.imshow(opencv2matplotlib(m[0]))
            f = plt.figure(figsize=(11, 1 + nmembers))
            # nx = int(f.get_figwidth() * f.dpi)
            # ny = int(f.get_figheight() * f.dpi)
            f.figimage(im)
            # plt.show()
            plt.tight_layout()
            plt.savefig(dir_fid.as_posix() + "_motage.pdf")
