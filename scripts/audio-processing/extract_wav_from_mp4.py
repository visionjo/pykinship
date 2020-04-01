# AudioExtract.py
import glob

import moviepy.editor as mp
from pathlib import Path
from tqdm import tqdm

dir_data = '../../data/fiw-videos/new-processed/'
f_scenes = glob.glob(f"{dir_data}*/*/scenes/*.mp4")

dirs_out = [Path(f).parent.joinpath('audio') for f in f_scenes]
_ = [d.mkdir(exist_ok=True) for d in dirs_out]

for f_scene in tqdm(f_scenes):
    fout = f_scene.replace('scenes/', 'scenes/audio/').replace('.mp4', '.wav')
    if not Path(fout).is_file():
        try:
            clip = mp.VideoFileClip(f_scene)

            clip.audio.write_audiofile(fout)
        except OSError:
            print(f_scene)
