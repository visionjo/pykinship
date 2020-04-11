import shutil
import sys
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

if not '../../src' in sys.path:
    sys.path.append('../..')
import numpy as np
import pandas as pd
from pathlib import Path

import shutil


def make_dirs(li_dirs):
    [Path(d).mkdir(exist_ok=True, parents=True) for d in li_dirs]


dir_data = '../data/fiw-videos/new-processed/'  # '/Users/jrobby/data/FIDs/'
# df = read_family_member_list(f_urls)
fn_encodings = 'encodings.pkl'
f_master_list = '../data/fiw-videos/fiw-videos-mid-master.pkl'
dir_fids = glob(dir_data + 'F????/')
dir_fids.sort()
df = pd.read_pickle(f_master_list)
for dir_fid in tqdm(dir_fids):
    dir_mids = glob(dir_fid + 'MID*/')

    fid = dir_fid.split('/')[-2]
    df_cur = df.loc[df.fid == fid]
    df_cur = df_cur.explode('mid').reset_index(drop=True)
    for dir_mid in dir_mids:
        print(dir_mid)
        Path(dir_mid).joinpath('shot-thumbnails').mkdir(exist_ok=True)
        mid = int(dir_mid.replace('/', '').split('MID')[-1])
        df_self = df_cur.loc[df_cur.mid == mid]
        encodings_mid = pd.read_pickle(dir_mid + 'encodings.pkl')
        if not len(encodings_mid):
            continue
        features_mid = list(encodings_mid.values())
        li_vids = df_self.vid.to_list()

        for vid in li_vids:
            f_encodings = dir_fid + vid + '/scenes/encodings.pkl'
            if Path(f_encodings).is_file():
                Path(dir_mid).joinpath('shot-thumbnails').joinpath(vid).mkdir(exist_ok=True)
                encodings_vid = pd.read_pickle(f_encodings)
                encodings_vid_tup = tuple(encodings_vid.items())

                for f, en in encodings_vid.items():

                    cs = cosine_similarity(features_mid, en.reshape(1, -1))
                    if np.median(cs) > .4:
                        fin = dir_data + f.replace(vid + '/', vid + '/scenes/cropped/') + '.jpg'
                        fout = f"{dir_mid}shot-thumbnails/" + vid + '/' + f.split('/')[-1] + ".jpg"
                        shutil.copy(fin, fout)
            # features_vid = [e[1] for e in encodings_vid_tup]
            # cs = cosine_similarity(features_vid, features_mid)
            # scores = np.median(cs, axis=1)
            # ffiles = np.array([e[0].replace(vid + '/', vid + '/scenes/cropped/') + '.jpg' for e in encodings_vid_tup])
            # matches = ffiles[scores > 0.23]
            # _ = [shutil.copy(f"{dir_data}{f}", f"{dir_mid}shot-thumbnails/" + f.split('/')[-1]) for f in ffiles]
            # print(scores)
