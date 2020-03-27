# AudioExtract.py
import glob
from pathlib import Path

import moviepy.editor as mp
from tqdm import tqdm

dir_data = '/Volumes/MyWorld/FIW_Video/data/processed/'
f_scenes = glob.glob(f"{dir_data}*/*/scenes/*.mp4")

dirs_out = [Path(f).parent.joinpath('audio') for f in f_scenes]
_ = [d.mkdir(exist_ok=True) for d in dirs_out]

for f_scene in tqdm(f_scenes):
    clip = mp.VideoFileClip(f_scene)
    fout = f_scene.replace('scenes/', 'scenes/audio/').replace('.mp4', '.wav')
    clip.audio.write_audiofile(fout)
