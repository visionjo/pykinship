# AudioExtract.py
import glob
from pathlib import Path

import moviepy.editor as mp
from tqdm import tqdm

dir_data = Path("../../data/fiw-mm/VIDs-aligned-tp/")
dir_out = Path("../../data/fiw-mm/VIDs-aligned-wav/")
f_scenes = dir_data.rglob("*.mp4")

for f_scene in tqdm(f_scenes):
    fout = Path(str(f_scene).replace(str(dir_data), str(dir_out))).with_suffix(".wav")
    fout.parent.mkdir(exist_ok=True, parents=True)
    if not fout.is_file():
        # try:
        clip = mp.VideoFileClip(str(f_scene))
        clip.audio.write_audiofile(str(fout))
        # except OSError:
        #     print(f_scene)
