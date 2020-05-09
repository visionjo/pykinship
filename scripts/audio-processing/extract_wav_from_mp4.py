# AudioExtract.py
import glob
from pathlib import Path

import moviepy.editor as mp
from tqdm import tqdm

dir_data = Path("/Volumes/MyWorld/FIW-MM/raw/")
dir_out = dir_data / "audio"
f_videos = glob.glob(f"{dir_data}/F????/v?????/*.mp4")

# dirs_out = [Path(f).parent.joinpath("audio") for f in f_scenes]
# _ = [d.mkdir(exist_ok=True) for d in dirs_out]

for f_video in tqdm(f_videos):
    vid = Path(f_video).parent.name
    fout = dir_out / Path(vid).with_suffix(".wav")
    if fout.is_file():
        continue
    print(f_video)
    clip = mp.VideoFileClip(f_video)
    clip.audio.write_audiofile(fout)
