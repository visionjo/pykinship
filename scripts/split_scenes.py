import glob
import os

from swifter import swifter

ncpus = swifter.cpu_count()
import pandas as pd
from pathlib import Path

data_dir = "../data/fiw-videos/new-raw/"
data_out = "../data/fiw-videos/interm/"

cmd = "scenedetect --input {} --output {} --stats stats.csv detect-content save-images  export-html list-scenes  split-video --filename {}   --copy"

dirs_in = [d + '/' for d in glob.glob(data_dir + '*') if Path(d).is_dir()]


# for din in reversed(dirs_in):
def split_videos(din):
    din = din['path']
    try:
        Path(din.replace('raw', 'processed')).mkdir()
    except:
        print(Path(din.replace('raw', 'processed')), 'exists')
        return
    fid = din.split('/')[-2]
    print(fid)
    vfiles = glob.glob(din + '*/*.mp4')
    for vfile in vfiles:
        dout = "/".join(vfile.split('/')[:-1]) + '/scenes'
        dout = dout.replace('raw', 'processed')
        # print(vfile, dout)
        # if Path(dout).exists():
        #     continue
        os.system(cmd.format(f"'{vfile}'", dout, dout.split('/')[-2]))


df = pd.DataFrame(dirs_in, columns=['path'])

df.swifter.apply(lambda x: split_videos(x), axis=1)
