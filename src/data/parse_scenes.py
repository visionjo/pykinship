from glob import glob

import numpy as np
import pandas as pd
import shutil
from pathlib import Path

from src.data.process_videos import meta_face_location_to_bb


#
# def process_frames():
#     input_movie = cv2.VideoCapture("tbbt.mp4")
#     length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Load some sample pictures and learn how to recognize them.
#     lmm_image = face_recognition.load_image_file("sheldon.jpg")
#     lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]
#
#     al_image = face_recognition.load_image_file("penny.jpg")
#     al_face_encoding = face_recognition.face_encodings(al_image)[0]
#
#     known_faces = [
#         lmm_face_encoding,
#         al_face_encoding
#     ]
#     # Initialize some variables
#     face_locations = []
#     face_encodings = []
#     face_names = []
#     frame_number = 0
#
#     current_path = os.getcwd()
#
#     counter = 0
#     counter1 = 0
#
#     while True:
#         # Grab a single frame of video
#         ret, frame = input_movie.read()
#         frame_number += 1
#
#         # Quit when the input video file ends
#         if not ret:
#             break
#
#         # Find all the faces and face encodings in the current frame of video
#         face_locations = face_recognition.face_locations(frame)
#         face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#         face_names = []
#         for face_encoding in face_encodings:
#             # See if the face is a match for the known face(s)
#             match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
#
#             # If you had more than 2 faces, you could make this logic a lot prettier
#             # but I kept it simple for the demo
#             name = None
#             if match[0]:
#                 name = "Sheldon Cooper"
#             elif match[1]:
#                 name = "Penny"
#
#             face_names.append(name)
#
#         # Label the results
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             if not name:
#                 continue
#
#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#             crop_img = frame[top:bottom, left:right]
#             if (name == "Sheldon Cooper"):
#                 cv2.imwrite(current_path + "/face_database/Sheldon/" + "sheldon" + str(counter) + ".png", crop_img)
#                 counter = counter + 1
#             elif (name == "Penny"):
#                 cv2.imwrite(current_path + "/face_database/Penny/" + "penny" + str(counter1) + ".png", crop_img)
#                 counter1 = counter1 + 1
#
#         # Write the resulting image to the output video file
#         print("Writing frame {} / {}".format(frame_number, length))

# # All done!
# input_movie.release()
# cv2.destroyAllWindows()
# create_csv.CreateCsv(current_path + "/face_database/")

def make_dirs(li_dirs):
    [Path(d).mkdir(exist_ok=True) for d in li_dirs]


dir_fids = '/home/jrobby/master-version/fiwdb/FIDs/'  # '/Users/jrobby/data/FIDs/'
# df = read_family_member_list(f_urls)

procese_shots = True

dir_root = '/Volumes/MySpace/scenes/'

imimages = glob(dir_root + '*/scenes/*.jpg')
imimages.sort()
dirs_scenes = [d for d in Path(dir_root).glob('*') if Path(d).is_dir()]

dir_faces = [str(d) + '/scenes/faces/' for d in dirs_scenes if Path(str(d) + '/scenes/faces/').is_dir()]
dir_scenes_parsed = [f.replace('/scenes/faces/', '/parse_scenes/') for f in dir_faces]
make_dirs(dir_scenes_parsed)
# for dir_face in dir_faces:
# dir_face = dir_faces[0]

files_bb = np.array(glob(dir_face + '*-meta.json'))
jsonΩ
files_bb.sort()

shot_ids = np.unique([f.split('-')[-4] for f in files_bb])
shot_ids.sort()
arr_lut = np.array([f.split('-')[-4] for f in files_bb])
for shot_id in shot_ids:
    ids = shot_id == arr_lut

    f_cur_shot = list(files_bb[ids])
    # f_cur_shot = f_cur_shot if len(f_cur_shot) > 1 else [f_cur_shot]
    # for f_cur in f_cur_shot:
    #     f_prediction
    meta = [meta_face_location_to_bb(f) for f in f_cur_shot]
    predictions = [np.loadtxt(f.replace('-meta.json', '-predictions.csv'), delimiter=',').astype(bool).flatten() for
                   f
                   in
                   f_cur_shot]
    npredictions = len(predictions)
    print(len(predictions), predictions)

    if np.any(predictions):
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
