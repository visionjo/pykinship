from pathlib import Path

import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

FPS = 25.0
DIGITS = 3
path_dtrack = Path("../../tracks").resolve()

path_vids = Path("/Volumes/MyWorld/FIW-MM/VIDs")
path_out = Path("/Volumes/MyWorld/FIW-MM/VIDs-aligned")
path_out.mkdir(exist_ok=True)
path_tracks = list(path_dtrack.rglob("*.pkl"))

path_tracks.sort()
print(len(path_tracks))

for path_track in tqdm(path_tracks, total=len(path_tracks), unit="files"):
    tracks = pd.read_pickle(path_track)
    ntracks = len(tracks)
    print(f"There are {ntracks} in {path_track.name}")

    if not ntracks:
        continue
    f_source = path_vids / path_track.with_suffix(".mp4").name
    d_out = path_out / path_track.with_suffix("").name
    try:
        d_out.mkdir()
    except Exception:
        print("skipping", path_track)
        continue
    for k, track in enumerate(tracks):
        print(f"{k}/{ntracks}")
        track = track["track"]
        start_frame, end_frame = track["frame"][0], track["frame"][-1]
        start_time, end_time = start_frame / FPS, end_frame / FPS
        # int to string and add leading zeros
        fout = d_out / f"track-{str(k).zfill(DIGITS)}.mp4"
        ffmpeg_extract_subclip(f_source, start_time, end_time, targetname=f"{fout}")
