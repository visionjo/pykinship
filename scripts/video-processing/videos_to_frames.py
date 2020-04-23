from pathlib import Path
from os import system

root = Path("../../data/fiw-mm")
data = Path(f"{root}/FIDs-MM")
image = Path(f"{data}/visual/image")
video = Path(f"{data}/visual/video")
path_out = Path(f"{data}/visual/video-frames")

command = "ffmpeg -i {} -vf  fps=25  {}/frame-%03d.png"

for f_clip in video.glob("F????/MID*/*.mp4"):
    dout = Path(str(f_clip).replace(str(video), str(path_out))).with_suffix("")
    dout.mkdir(parents=True, exist_ok=True)
    system(command.format(f_clip, dout))
