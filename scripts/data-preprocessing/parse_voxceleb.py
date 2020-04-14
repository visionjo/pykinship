import glob
import os
from pathlib import Path

import pandas as pd

# import swifter

# from swifter import swifter as sw
# print(str(swifter.__version__))
# import re
#
# import unicode
#
#
# def slugify(value):
#     """
#     Normalizes string, converts to lowercase, removes non-alpha characters,
#     and converts spaces to hyphens.
#     """
#     import unicodedata
#     value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
#     value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
#     value = unicode(re.sub('[-\s]+', '-', value))
#     # ...
#     return value
#
#
# import string
#
# valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
# '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# filename = "This Is a (valid) - filename%$&$ .txt"
#
# for filename in vfiles:
#     fout = ''.join(c for c in Path(filename).name if c in valid_chars)
#     fout = f"{Path(filename).parent}/{fout}"
#     if filename == fout:
#         continue
#     shutil.move(filename, fout)
#
# 'This Is a (valid) - filename .txt'
dir_wav = Path('/Volumes/MyWorld/FIW-MM/wav-mm/')
frame_rate = 25

video_paths = [v for v in dir_wav.glob('F????/MID*/*/*/*.mp4') if not 'clips' in str(v)]
video_paths.sort()
# vfiles.sort()
tfiles = [p.parent.parent / (p.parent.name + '.txt') for p in video_paths]

# # vfiles = [v for v in vfiles if not 'clip' in v]
# tfiles = glob.glob(f'{dir_wav}F????/MID*/*/*.txt')
# tfiles.sort()

print(tfiles)
for tfile in tfiles:
    print(tfile)

    # f_meta = str(Path(tfile).parent) + '.txt'
    # with open(tfile, 'r') as f:
    #     content = f.readlines()
    meta = [c.strip().split() for c in tfile.read_text().splitlines()]
    # meta = [c.rstrip().strip().split() for c in content]

    youtube_id = meta[1][-1]
    print(meta)
    df = pd.DataFrame(meta[5:], columns=['ref', 'start', 'end'])
    df['duration'] = df['end'].astype(float) - df['start'].astype(float)

    obin = dir_wav + "/".join(tfile.replace(dir_wav, '').split('/')[:2]) + '/clips/' + youtube_id + '/'
    print(obin)
    Path(obin).mkdir(exist_ok=True)

    # df.head()

    nclips = len(df)
    vfile = tfile.replace('.txt', '/')
    vfile = glob.glob(vfile + '*.mp4')
    if not vfile:
        continue
    vfile = vfile[0]
    for i in range(nclips):

        # vfile = vfiles[0]
        ofile = obin + str(df.iloc[i]['ref'].split('/')[-1]).split("_")[-1] + ".mp4"
        if Path(ofile).is_file():
            continue

        Path(ofile).parent.mkdir(exist_ok=True, parents=True)
        start, span = float(df.iloc[i]['start']), df.iloc[i]['duration']
        print(ofile)
        var = r"\ "
        vfile = vfile.replace(" ", var)
        os.system(
            f'ffmpeg -i {vfile} -ss {start} -threads 16 -t {span} -r {frame_rate} -async 1 -qscale:v 5 -deinterlace -vf scale=-1:360  {ofile}')
