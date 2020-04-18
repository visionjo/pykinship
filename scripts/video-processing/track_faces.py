from pathlib import Path
from mtcnn import MTCNN
import cv2
import pickle

path_data = Path('../../data/fiw-mm/').resolve()
path_images = path_data / 'FIDs-MM/visual/image'
path_videos = path_data / 'FIDs-MM/visual/video'
path_encodings = path_data / 'features/image/arcface'
path_out = path_data / 'features/visual/video'

last_fid_mid = None
detector = MTCNN()
for v_file in path_videos.rglob('*.mp4'):
    # each video found in nested directories
    print(v_file)

    # fid_mid = str(v_file.parent).replace(f"{path_videos}/", '')

    # if last_fid_mid != fid_mid:
    #     # load image features of FID.MID first time loading on of their videos
    #     path_mid_encodings = path_encodings / fid_mid / 'encodings.pkl'
    #     with open(path_mid_encodings, 'rb') as f:
    #         encodings = pickle.load(f)
    #         features = np.array(list(encodings.values()))
    #         del encodings

    video = cv2.VideoCapture(str(v_file))
    imdetections = {}

    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ref = Path(str(v_file).replace(str(path_videos), '')).with_suffix('')
            detections = detector.detect_faces(frame)
            imdetections[str(ref)] = detections

    # with open(f"{path_out}mtcnn_detections.pkl", 'wb') as f:
    #     pickle.dump(imdetections, f)

# :
# # Our operations on the frame come here
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# except EOFError:
# # When everything done, release the capture
# video.release()
