import glob
import shutil
from pathlib import Path

from src.data.videos import get_video_metadata


def bb_intersection_over_union(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # compute IoU by taking the intersection area and dividing it by the sum of BBs A + B areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def rename_by_convention(dir_in):
    all_files = glob.glob(dir_in + '*/fr*')
    for afile in all_files:
        ofile = afile.split('/')[-1]

        fr_id = ofile.split('fr')[-1].split('_')[0]
        fr_id = "%.6d" % int(fr_id)

        face_id = ofile.split('face')[-1].split('.')[0].replace('-encoding', '').replace('-predictions', '').replace(
            '-meta', '')
        face_id = "%.2d" % int(face_id)

        ext = ofile.split('.')[-1]

        dout = "/".join(afile.split('/')[:-1])

        fout = f"{dout}/fr{fr_id}_face{face_id}.{ext}"
        # print(afile, fout)
        shutil.move(afile, fout)


def get_face_ids_from_image_list(li_impaths):
    ids = list(set([f.split('.')[-2].split('face')[-1] for f in li_impaths]))
    ids.sort()
    return ids


dir_data = '/Volumes/MySpace/kinship/'
dir_video = f'{dir_data}ben_affleck.ben/'
f_video = '/Volumes/MySpace/kinship/ben_affleck.ben/ben_affleck.ben.mp4'
meta = get_video_metadata(f_video)

dir_predictions = f'{dir_video}predictions/'
dir_meta = f'{dir_video}meta/'
dir_encodings = f'{dir_video}encodings/'
dir_faces = f'{dir_video}faces/'

f_metas = glob.glob(dir_meta + '*.json')
ids = get_face_ids_from_image_list(f_metas)

i = "face" + ids[0]
cfaces = [f for f in f_metas if f.count(i)]

import pandas as pd

nframes = meta['frame_count']

import numpy as np

cur_track = 0

ids = [int(i) for i in ids]
# for k, cface in enumerate(cfaces):
li_track_ids = []

for i in ids:
    track_ids = -1 * np.ones((nframes,))
    box_prev = 0
    for k in range(nframes):
        fin = '{0}meta/fr{1:06d}_face{2:02d}.json'.format(dir_video, k, i)

        if Path(fin).is_file():
            df_meta = pd.read_json(fin)
            df_meta.columns = df_meta.iloc[0]
            df_meta = df_meta.drop(0)
            df_meta.reset_index(inplace=True)
            l, t, r, b = tuple(df_meta.iloc[0].bb)
            box_cur = l, t, r, b
            if box_prev:
                iou = bb_intersection_over_union(box_cur, box_prev)

                track_ids[k] = iou

                # if iou < .75:
                #     cur_track += 1
                # track_ids[k][1] = cur_track

            box_prev = box_cur
        else:
            track_ids[k] = 0
    li_track_ids.append(track_ids)

df_meta.iloc[0] = df_meta.iloc[1]

do_renaming = False

# meta = get_video_metadata(f_video)
# video = cv2.VideoCapture(f_video)

all_faces = glob.glob(f"{dir_faces}fr*")
all_faces.sort()

ids = get_face_ids_from_image_list(all_faces)
for i in ids:
    f_faces = glob.glob(f"{dir_faces}fr*{i}.png")
for frame_id in range(meta['frame_count']):
    f_face = f"{dir_faces}fr{frame_id}*"
    file_faces = glob.glob(f_face + '_face*')

    for j in range(len(file_faces)):
        face_id = file_faces[j].split('.')[-2].split('face')[-1]
        print(face_id)

subjects = ['dan_gronkowski.rob', 'muhammad_ali.muhammad', 'hafez_al-assad.bashar', 'diane_gronkowski.rob',
            'chris_gronkowski.rob', 'gord_gronkowski.rob', 'gordie_gronkowski.rob', 'glenn_gronkowski.rob',
            'dan_gronkowski.rob', 'rob_gronkowski.rob', 'casey_affleck.ben',
            'ben_affleck.ben']  # , 'deborah_phelps.michael']

if do_renaming:
    for subject in subjects:
        print(subject)
        dir_video = f'{dir_data}{subject}/'
        rename_by_convention(dir_video)
