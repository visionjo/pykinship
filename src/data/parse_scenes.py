from glob import glob

import shutil
import sys

if not '../../src' in sys.path:
    sys.path.append('../..')
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.process_videos import meta_face_location_to_bb, bb_intersection_over_union


def make_dirs(li_dirs):
    [Path(d).mkdir(exist_ok=True, parents=True) for d in li_dirs]


dir_fids = '/home/jrobby/master-version/fiwdb/FIDs/'  # '/Users/jrobby/data/FIDs/'
# df = read_family_member_list(f_urls)

dir_root = '/Volumes/MySpace/kinship/'

imimages = glob(dir_root + '*/scenes/faces/*.png')
imimages.sort()

dirs_scenes = [str(d) for d in Path(dir_root).glob('*') if Path(d).is_dir()]

dir_faces = [str(d) + '/scenes/faces/' for d in dirs_scenes if Path(str(d) + '/scenes/faces/').is_dir()]
dir_scenes_parsed = [f.replace('/scenes/faces', '/scenes_parsed/') for f in dir_faces]
make_dirs(dir_scenes_parsed)


def copy_matched_data(files_in, files_out):
    # print(files_out)
    [shutil.copy(fin.replace('-meta.json', '.png'), fout) for fin, fout in zip(files_in, files_out)]
    fout = "-".join(files_out[0].split('-')[:-1]) + '-bb.csv'
    pd.DataFrame(meta).to_csv(fout, header=False, index=False)


for dir_face in dir_faces:
    files_bb = np.array(glob(dir_face + '*-meta.json'))
    files_bb.sort()

    shot_ids = np.unique([f.split('-')[-4] for f in files_bb])
    shot_ids.sort()
    arr_lut = np.array([f.split('-')[-4] for f in files_bb])
    print(arr_lut)
    for shot_id in shot_ids:
        ids = shot_id == arr_lut

        f_cur_shot = np.array(list(files_bb[ids]))
        meta = {f.split('/')[-1]: meta_face_location_to_bb(f) for f in f_cur_shot}
        meta = {k: v for k, v in meta.items() if v}

        predictions = {f.split('/')[-1]: np.loadtxt(f.replace('-meta.json', '-predictions.csv'), delimiter=',').astype(bool).flatten() for f in f_cur_shot}

        npredictions = len(predictions)
        print(len(predictions), predictions)

        if not np.any(predictions):
            # case 1, no True predictions in shot
            continue

        print(f_cur_shot)

        # case 1, single face in all 3 snapshots of shot
        if npredictions == 3 and False:
            if np.all(predictions):
                fouts = [f.replace('-meta.json', '.png') for f in f_cur_shot]
                print(fouts)
                [shutil.copy(f, f.replace('/scenes/faces/', '/parse_scenes/'))
                 for f in fouts]
                fout = "-".join(
                    f_cur_shot[0].replace('/scenes/faces/', '/parse_scenes/').split('-')[:-1]) + '-bb.csv'
                pd.DataFrame(meta).to_csv(fout, header=False, index=False)
                # [shutil.copy(f, )
                #  for f in fouts]
            else:
                print('no prediction for {}'.format(shot_id))
        elif np.any(np.sum(predictions, axis=1) == 3):
            pass
