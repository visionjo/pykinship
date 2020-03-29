# AudioExtract.py
import glob

import moviepy.editor as me
import multiprocessing as mp
from pathlib import Path
from time import time

overwrite = False


def save_audio(f):
    try:
        clip = me.VideoFileClip(f[0])
        clip.audio.write_audiofile(f[1])

    except OSError:
        print(f)


# Step 1: Init multiprocessing.Pool()
ncpu = mp.cpu_count()

dir_data = '../data/fiw-videos/new-processed/'
f_scenes = glob.glob(f"{dir_data}*/*/scenes/*.mp4")

if not overwrite:
    f_scenes = [f for f in f_scenes if not Path(f.replace('scenes/', 'scenes/audio/').replace('.mp4', '.wav')).is_file()]
dirs_out = [Path(f).parent.joinpath('audio') for f in f_scenes]
_ = [d.mkdir(exist_ok=True) for d in dirs_out]

start = time()
while len(f_scenes) > ncpu:
    fin = f_scenes[:ncpu]

    args = [(f, f.replace('scenes/', 'scenes/audio/').replace('.mp4', '.wav')) for f in fin]

    with mp.Pool(ncpu) as pool:
        pool.map(save_audio, args)
    del f_scenes[:ncpu]
else:
    fin = f_scenes
    args = [(f, f.replace('scenes/', 'scenes/audio/').replace('.mp4', '.wav')) for f in fin]

    with mp.Pool(ncpu) as pool:
        pool.map(save_audio, args)

end = time()

print(end - start)
# 121.23533892631531
