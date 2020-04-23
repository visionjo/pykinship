import subprocess
from pathlib import Path
from tqdm import tqdm

wild_flag = "F????/v?????/scenes/mp4/*.mp4"
path_videos = Path("/Users/jrobby/GitHub/pykinship/data/interm/")
path_out = Path("/Volumes/MyWorld/FIW-MM/clips")
command = "ffmpeg -i {} -c copy -an {}"
#

for path_video in tqdm(reversed(list(path_videos.glob(wild_flag)))):
    f_out = path_out.joinpath(
        str(path_video).replace(f"{str(path_videos)}/", "").replace("scenes/mp4/", "")
    )
    # print(path_video, f_out))
    f_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.call(command.format(str(path_video), str(f_out)), shell=True)
