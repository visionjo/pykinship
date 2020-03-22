import glob

import argparse
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from src.data.videos import print_video_metadata, get_video_metadata

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description="FaceTracker")
parser.add_argument('--data_dir', type=str, default='data/work', help='Output direcotry')
parser.add_argument('--videofile', type=str, default='', help='Input video file')
parser.add_argument('--reference', type=str, default='', help='Name of the video')
parser.add_argument('--crop_scale', type=float, default=0.5, help='Scale bounding box')
parser.add_argument('--min_track', type=int, default=100, help='Minimum facetrack duration')
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25, help='Number of missed detections allowed')
parser.add_argument('--min_face_size', type=float, default=0.03, help='Minimum size of faces')
opt = parser.parse_args()

setattr(opt, 'avi_dir', str(Path(opt.data_dir).joinpath('pyavi')))
setattr(opt, 'tmp_dir', str(Path(opt.data_dir).joinpath('pytmp')))
setattr(opt, 'work_dir', str(Path(opt.data_dir).joinpath('pywork')))
setattr(opt, 'crop_dir', str(Path(opt.data_dir).joinpath('pycrop')))
setattr(opt, 'frames_dir', str(Path(opt.data_dir).joinpath('pyframes')))


# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========
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


def check_if_renamed(dirin):
    f_faces = [str(f).split('/')[-1] for f in Path(dirin).joinpath('faces').glob('*.png')]
    nchars = np.array([len(f) for f in f_faces])
    return np.all(nchars == 19)


def rename_by_convention(dir_in):
    if check_if_renamed(dir_in):
        return
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


def percent_correct_match(din, fout, nframes):
    """
    :param din directory containing bb information (json)
    :param fout pickle file path to save array of iou values (K x F), where is K is number of frames and F is the max
                number of faces in a frame at a given instance.
    """

    track_ids = -1 * np.ones((nframes,))
    li_histo_differences = []
    hist_prev = 0
    for k in range(nframes):
        fin = '{0}fr{1:06d}_face{2:02d}.png'.format(din, k, n)

        if Path(fin).is_file():
            df_meta = cv2.imread(fin, )
            if box_prev:
                hist_cur = None

                track_ids[k] = None

            # if iou < .75:
            #     cur_track += 1
            # track_ids[k][1] = cur_track

            box_prev = hist_cur
        # else:
        #     box_prev = -1
    li_histo_differences.append(track_ids)
    pd.to_pickle(li_histo_differences, fout)


# ========== ========== ========== ==========
# # Color Histogram
# ========== ========== ========== ==========
def calculate_color_histogram_diff_neighboring_frames(din, fout, nframes):
    """
    :param din directory containing bb information (json)
    :param fout pickle file path to save array of iou values (K x F), where is K is number of frames and F is the max
                number of faces in a frame at a given instance.
    """
    li_histo_differences = []
    for n in ids:
        track_ids = -1 * np.ones((nframes,))
        hist_prev = 0
        for k in range(nframes):
            fin = '{0}fr{1:06d}_face{2:02d}.png'.format(din, k, n)

            if Path(fin).is_file():
                df_meta = cv2.imread(fin, )
                if box_prev:
                    iou = bb_intersection_over_union(box_cur, box_prev)

                    track_ids[k] = iou

                # if iou < .75:
                #     cur_track += 1
                # track_ids[k][1] = cur_track

                box_prev = hist_cur
            # else:
            #     box_prev = -1
        li_histo_differences.append(track_ids)
    pd.to_pickle(li_histo_differences, fout)


def meta_face_location_to_bb(f_json):
    try:
        df_meta = pd.read_json(f_json)
        l, t, r, b = tuple(df_meta.iloc[1][2])
        bb = l, t, r, b
    except:
        bb = None
    return bb


def calculate_iou_bb_neighboring_frames(din, fout, nframes):
    """
    :param din directory containing bb information (json)
    :param fout pickle file path to save array of iou values (K x F), where is K is number of frames and F is the max
                number of faces in a frame at a given instance.
    """
    f_metas = glob.glob(din + "*.json")
    ids = get_face_ids_from_image_list(f_metas)

    i = "face" + ids[0]
    # cfaces = [f for f in f_metas if f.count(i)]
    # nframes = meta['frame_count']
    # cur_track = 0

    ids = [int(i) for i in ids]

    li_track_ids = []
    for n in ids:
        track_ids = -1 * np.ones((nframes,))
        box_prev = 0
        for k in range(nframes):
            fin = '{0}fr{1:06d}_face{2:02d}.json'.format(din, k, n)

            if Path(fin).is_file():
                box_cur = meta_face_location_to_bb(fin)
                if box_prev:
                    iou = bb_intersection_over_union(box_cur, box_prev)

                    track_ids[k] = iou

                # if iou < .75:
                #     cur_track += 1
                # track_ids[k][1] = cur_track

                box_prev = box_cur
            # else:
            #     box_prev = -1
        li_track_ids.append(track_ids)
    pd.to_pickle(li_track_ids, fout)


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

nframes = meta['frame_count']

cur_track = 0

    ids = [int(i) for i in ids]
    # for k, cface in enumerate(cfaces):

    if do_predictions:
        # dins = [d + '/' for d in glob.glob(f'{dir_data}/*/predictions') if Path(d).is_dir()]
        # for dir_video in dins:
        #     pass

        # f_prediction = list(Path(dir_video).joinpath('predictions').glob('*.csv'))[0]
        # sample = pd.read_csv(f_prediction)
        # nscores = len(sample)
        li_score_arrs = []
        for i in ids:
            score_arr = np.zeros((nframes, 2))
            f_predictions = Path(dir_video).joinpath('predictions').glob('*{:02d}.csv'.format(i))
            for f_prediction in f_predictions:
                sample = pd.read_csv(f_prediction).T.values
                k = int(str(f_prediction).split("/fr")[-1].split('_')[0])
                score_arr[k][0] = (sample.sum() / sample.size)
                score_arr[k][1] = (sample.size - sample.sum()) / (sample.size + sample.sum())
            li_score_arrs.append(score_arr)
        pd.to_pickle(li_score_arrs, dir_video + 'predictions.pkl')

        ids = [np.where(arr[:, 0])[0] for arr in li_score_arrs]
        li = []
        for idz in ids:
            tmp = np.zeros_like(idz)
            for i in range(len(idz[:-1])):
                tmp[i] = idz[i + 1] - idz[i] - 1
            li.append(tmp == 0)

        indices = []
        for idz, l in zip(ids, li):
            indices.append(idz[l])
        dout = dir_video + 'parsed_frames/'
        Path(dout).mkdir(exist_ok=True)
        dir_faces = f"{dir_video}faces/"
        for i, index in enumerate(indices):
            dout1 = '{0}f{1:02d}/'.format(dout, i)
            Path(dout1).mkdir(exist_ok=True)
            prev_id = 0
            counter = 0
            dout2 = "{0}t{1:04d}/".format(dout1, counter)
            for j in index:
                diff = j - prev_id - 1
                if diff:
                    dout2 = "{0}t{1:04d}/".format(dout1, counter)
                    Path(dout2).mkdir(exist_ok=True)
                    counter += 1
                fout = '{0}fr{1:06d}_face{2:02d}.png'.format(dout2, j, i)
                fin = fout.replace(dout2, dir_faces)
                shutil.copy(fin, fout)
                prev_id = j

    if do_iou:
        dins = [d + '/' for d in glob.glob(f'{dir_data}/*/meta') if Path(d).is_dir()]
        for dir_video in dins:
            calculate_iou_bb_neighboring_frames(dir_video, dir_video + 'iou.pkl', nframes)

    if do_histodiff:
        dins = [d + '/' for d in glob.glob(f'{dir_data}/*/predictions') if Path(d).is_dir()]
        # for dir_video in dins:
        #     fout = dir_video + 'id_scores_arr.pkl'
        #
        #     f_pr

        if do_renaming:
            dirs = [d for d in glob.glob(dir_data + '*/predictions/')]

            subjects = [d.replace('predictions', '').replace('//', '/').split('/')[-1] for d in dirs]
            if not subjects or not len(subjects[0]):
                subjects = [d.replace('predictions', '').replace('//', '/').split('/')[-2] for d in dirs]

            # for d in glob.glob(dir_data + '*/predictions/')]
            for subject in subjects:
                print(subject)
                dir_video = f'{dir_data}{subject}/'
                rename_by_convention(dir_video)

            if False:
                # df_meta.iloc[0] = df_meta.iloc[1]

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
