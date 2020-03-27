import glob
import os
from pathlib import Path

data_dir = "/Volumes/MyWorld/FIW_Video/data/raw/"
data_out = "/Volumes/MyWorld/FIW_Video/data/processed/"

cmd = "scenedetect --input {} --output {} --stats stats.csv detect-content save-images  export-html list-scenes  split-video --filename {}   --copy"

dirs_in = [d + '/' for d in glob.glob(data_dir + '*') if Path(d).is_dir()]

for din in reversed(dirs_in):

    subject = din.split('/')[-2]
    print(subject)
    vfiles = glob.glob(din + '*/*.mp4')
    # vfiles = "/".join(din.split('/')[:-1])
    for vfile in vfiles:
        dout = "/".join(vfile.split('/')[:-1]) + '/scenes'
        # vfile = vfile.replace(' ', '\ ')
        dout = dout.replace('raw', 'processed')
        print(vfile, dout)
        if Path(dout).exists():
            continue
        # Path(dout).mkdir()
        os.system(cmd.format(f"'{vfile}'", dout, subject, subject))
