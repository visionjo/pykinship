#!/usr/bin/env python
"""
Script to convert mp4 files (videos) to png (images)
"""

from os import system
from pathlib import Path

PATH_ROOT = Path("../../data/fiw-mm")
PATH_DATA = Path(f"{PATH_ROOT}/FIDs-MM")
PATH_IMAGE = Path(f"{PATH_DATA}/visual/image")
PATH_VIDEO = Path(f"{PATH_DATA}/visual/video")
PATH_OUT = Path(f"{PATH_DATA}/visual/video-frames")

CMD = "ffmpeg -i {} -vf  fps=25  {}/frame-%03d.png"

for f_clip in PATH_VIDEO.glob("F????/MID*/*.mp4"):
    dout = Path(str(f_clip).replace(str(PATH_VIDEO), str(PATH_OUT))).with_suffix("")
    dout.mkdir(parents=True, exist_ok=True)
    system(CMD.format(f_clip, dout))
