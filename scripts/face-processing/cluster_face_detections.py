# import the necessary packages
import glob
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import shutil
import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


def load_features_and_faces(files_feats):
    f_faces = np.array(
        [file.replace('-features/', '-faces/').replace('.pkl', '.jpg') for file in files_feats])

    # ids = np.where([1 if is_file(f) else 0 for f in files_faces])[0]

    features = OrderedDict({f: pd.read_pickle(f) for f in files_feats})
    faces = OrderedDict({f: cv2.imread(f) for f in f_faces})
    # faces = {f: cv2.imread(f) for f in files_faces[ids]}

    return features, faces


from pathlib import Path

dir_root = '/Users/jrobby/Dropbox/FIW_Video/data/parse_scenes/'
dirs_video = glob.glob(f"{dir_root}*/scenes/faces/")
dirs_video.sort()
dirs_out = [d.replace('scenes/faces/', 'clusters/').replace('parse_scenes', 'processed') for d in dirs_video if Path(d.replace('scenes/faces/', '')
                                                                                                                     .replace('parse_scenes',
                                                                                                                              'processed')).is_dir()]

[Path(d).mkdir(exist_ok=True) for d in dirs_out]

f_features = []
for dir_faces in dirs_faces:
    for d_fid in io.dir_list(dir_faces):
        print(d_fid)
        for f in glob.glob(d_fid + '/*.pkl'):
            f_features.append(f)

pd.set_option('display.max_columns', 100)

df_data = pd.DataFrame(data=None, columns=['path', 'fid', 'mid', 'faceid', 'is_face'])
df_data.path = f_features
df_data.path = df_data.path.str.replace('/home/jrobby/master-version/fiwdb/FIDs-features/', '')
df_data.fid = df_data.path.str.rsplit('/')
df_data.mid = df_data.fid.apply(lambda x: x[1])
df_data.faceid = df_data.fid.apply(lambda x: x[2])
df_data.fid = df_data.fid.apply(lambda x: x[0])
df_data = df_data.reset_index(drop=True)

df_data.loc[df_data.mid.str.count('MID') == 0, 'is_face'] = 0
df_data.loc[df_data.mid.str.count('MID') == 1, 'is_face'] = 1

fids = df_data.fid.unique()
for fid in fids:
    print(fid)
    ids = df_data.fid == fid
    df = df_data.loc[ids]
    feats = df[['mid', 'path', 'faceid']].groupby('mid').apply(
        lambda x: [(xxx, pd.read_pickle(CONFIGS.path.dfeatures + xx)) for xx, xxx in zip(x['path'], x['faceid'])]).to_dict()

    features = []
    label = []
    faceid = []
    for mid, feat in feats.items():
        for f in feat:
            label.append(mid)
            features.append(f[1])
            faceid.append(f[0])
    features_arr = np.array(features)
    label_arr = np.array(label)
    faceid = np.array(faceid)

    labels = np.unique(label_arr)
    labels.sort()
    label = labels[-1:]

    # add rectangles on the parameters of each class along the diagonal
    ids = np.where(label == label_arr)[0]

    feats2 = features_arr[ids]
    ids1 = np.where(label != label_arr)[0]

    allscores = []
    for lab in label_arr[ids]:
        feats1 = features_arr[label_arr == lab]
        l1 = label_arr[label_arr == lab]
        f1 = faceid[label_arr == lab]
        for j, feat in enumerate(feats1):
            scores = cosine_similarity(feat.reshape(1, -1), feats2)[0][:-1]
            if np.any(scores > .5):
                ids2 = np.where(scores > .5)[0]
                print(scores[ids2], l1[ids2], f1[ids2])
                # sc = scores[ids2[0]]
                # l = scores[ids2]
                # f = scores[ids2]
                # n = len(l1)
                # for i in range(n):
                #     print(sc[i],l[i],f[i])

        allscores.append(scores)

# dirs_faces = dirs_faces[-3:]
for (i, d_face) in tqdm.tqdm(enumerate(dirs_faces)):
    f_faces = [f for f in io.pklist(d_face + '*/*.pkl') if io.is_file(f.replace('-features', '').replace('.pkl', '.jpg'))]

    # [shutil.move(f, f.replace('.jpg', '') + '.jpg') for f in f_faces]
    print("[INFO] processing directory {}/{}".format(i + 1, len(f_faces)))

    print("[INFO] quantifying faces...")
    # dirs_subject_faces = [d + '/' for d in imagelist(dir_faces)]
    # dirs_subject_faces.sort()
    dir_out = dirs_faces[i].replace(dfeatures, dout)
    mkdir(dir_out)
    # for j, dir_subject_faces in enumerate(dirs_subject_faces):

    # print("[INFO] processing image {}/{}".format(j + 1, len(dir_subject_faces)))
    # dir_out = dir_subject_faces.replace(din, dout)
    # if not is_dir(dir_out):
    # f_faces = np.array(glob.glob(dir_subject_faces + '*.jpg'))
    if len(f_faces) == 0:
        continue
    f_faces.sort()
    try:
        features, faces = load_features_and_faces(f_faces)
    except:
        # print(dir_subject_faces)
        # print(dir_subject_faces)
        print(d_face)
        continue
    f_faces = np.array([f.replace(dfeatures, din).replace('.pkl', '.jpg') for f in f_faces])
    X = np.array(list(features.values()))
    db = DBSCAN(metric='cosine', n_jobs=2).fit(X)

    mkdir(dir_out)
    dir_outliers = dir_out + 'outlier/'

    tags = db.labels_
    # f_faces = np.array(list(faces.keys()))

    if np.any(tags == -1):
        mkdir(dir_outliers)
        ids = np.where(tags == -1)[0]
        fouts = f_faces[ids]
        success = [shutil.copy(f, dir_outliers + f.split('/')[-1]) for f in fouts]
        f_faces = f_faces[np.where(tags != -1)[0]]
        tags = tags[np.where(tags != -1)[0]]
    # if len(tags) == 0:
    #     continue
    utags = np.unique(tags)
    ntags = len(utags)
    if False:
        if ntags > 1:
            true_id = np.argmax(np.bincount(tags))
            ids = np.where(tags == true_id)[0]
            fouts = f_faces[ids]
            success = [shutil.copy(f, dir_out + f.split('/')[-1]) for f in fouts]
            f_faces = f_faces[np.where(tags != true_id)[0]]
            tags = tags[np.where(tags != true_id)[0]]

            for utag in np.unique(tags):
                odir = "{}/{}/".format(dir_out, utag)
                mkdir(odir)
                ids = np.where(tags == utag)[0]
                fouts = f_faces[ids]
                [shutil.copy(f, odir + f.split('/')[-1]) for f in fouts]
                # f_faces = f_faces[np.where(tags != utag)[0]]
                # tags = tags[np.where(tags != utag)[0]]
            print()
        elif ntags == 1:
            success = [shutil.copy(f, dir_out + f.split('/')[-1]) for f in f_faces]
    else:
        fouts = f_faces
        for utag in np.unique(tags):
            odir = "{}/{}/".format(dir_out, utag)
            mkdir(odir)
            ids = np.where(tags == utag)[0]
            fouts = f_faces[ids]
            [shutil.copy(f, odir + f.split('/')[-1]) for f in fouts]
