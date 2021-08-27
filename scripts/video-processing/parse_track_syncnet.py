import pandas as pd
import numpy as np
from pathlib import Path
from moviepy.video.io import VideoFileClip
from PIL import Image
from warnings import warn
from tqdm import tqdm

path_root = Path("/Volumes/MyWorld/FIW-MM/")
path_tracks = path_root / "tracks"
path_track_meta = path_root / "VIDs-aligned-tp"
path_out = path_root / "VIDs-aligned-tp-faces"
path_out.mkdir(exist_ok=True, parents=True)
path_videos = path_root / "VIDs-aligned"
vpaths = list(path_track_meta.rglob("*.mp4"))
vpaths.sort()

vids = np.unique([f.parent.name for f in vpaths if len(f.parent.name)])
vids.sort()

# vids = list(vids)[int(len(vids) / 2):]

for vid in tqdm((vids)):
    vid_tracks = [p for p in vpaths if p.parent.name == vid]
    if not vid_tracks:
        continue
    track_ids = [int(p.stem.split("-")[1]) for p in vid_tracks]
    path_track = path_tracks.joinpath(vid).with_suffix(".pkl")

    tracks = pd.read_pickle(path_track)
    vpath = path_videos.joinpath(vid).with_suffix(".mp4")
    counter = 0

    for track, track_id in zip(np.array(tracks)[track_ids], track_ids):
        vid_track = vid_tracks[counter]
        dout = Path(
            str(vid_track.parent / vid_track.stem).replace(
                str("VIDs-aligned-tp"), str("VIDs-aligned-tp-faces")
            )
        )
        print(dout)
        if dout.is_dir():
            continue
        dout.mkdir(exist_ok=True, parents=True)

        myclip = VideoFileClip.VideoFileClip(str(vid_track))

        bboxes = track["track"]["bbox"]
        for j, bbox in enumerate(bboxes):
            if j < myclip.duration:
                pil_image = Image.fromarray(myclip.get_frame(j))
                x1, y1, x2, y2 = bbox
                ## get the center and the radius
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2
                cr = max(w, h) // 2
                r = cr * 1.3
                bb = cx - r, cy - r, cx + r, cy + r

                pil_crop = pil_image.crop(bb)
                pil_crop.save(
                    dout.joinpath("face-" + str(j).zfill(3)).with_suffix(".jpg")
                )
            warn("Out of bounce")
        counter += 1
