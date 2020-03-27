from pathlib import Path

import pandas as pd
import wget
from pytube import YouTube

from src.data.videos import read_family_member_list

f_urls = "../data/family_members.csv"
d_out = "/Volumes/MyWorld/FIW_Video/data/raw/"
dir_scenes = '/home/jrobby/kinship/'
df = read_family_member_list(f_urls)
get_videos = False
cols = df.columns.to_list()
video_cols = [c for c in cols if c.count('video')]

df_videos = df[['ref'] + ['fid'] + ['mid'] + video_cols]


def process_subject(row, dir_out=d_out):
    ref = row['ref']
    fid = f"{row['fid']}.{row['mid']}"

    dout = Path(dir_out).joinpath(ref)
    with open(dout.joinpath('fiw.txt'), 'w') as f:
        f.write(fid)
    # del row['ref']
    vlist = [(k, v) for k, v in row.items() if pd.notna(v) if
             (k != 'ref') and (k != 'fid') and (k != 'mid')]
    for v in vlist:
        if Path(dout).joinpath(v[0]).is_dir():
            pass
            # print(dout.joinpath(v[0]))
        else:
            print(dout, v)
            do = str(Path(dout).joinpath(v[0])) + '/'
            url = v[1]

            Path(do).mkdir(parents=True, exist_ok=True)
            yt = YouTube(url)

            url_t = yt.thumbnail_url
            t = wget.download(url_t, out=do + 'thumbnail.png')
            describe = yt.description
            title = yt.title
            length = yt.length

            with open(do + 'meta.txt', 'w') as f:
                f.write(url + '/n/n')
                f.write(title + '/n')

                f.write("length: " + str(length))
                f.write("thumbnail url:" + url_t + '/n')
                f.write(describe + '/n/n')
            caption = yt.captions.get_by_language_code('en')

            if caption:
                cout = caption.generate_srt_captions()
                with open(do + 'caption.txt', 'w') as f:
                    f.write(cout)

            stream = yt.streams.first()
            stream.download(do)


df_videos.swifter.apply(lambda x: process_subject(x, d_out), axis=1)
# pd.swifter(df, process_subject)
# row = df_videos.iloc[0]
#
#
# Path()

# if get_videos:
#     fetch_videos(df, dir_out)
#
# # f_videos = Path(dir_videos).glob('*.mp4')
# # d_mid = Path(f_video).parent
# if process_videos:
#     nsubjects = len(df)
#     for i in range(nsubjects):
#         dir_mid = f"{dir_fids}{df['fid']}/MID{df['mid']}/"
#         encodings = encode_mids(dir_mid)
#         encodings = list(encodings.values())
#         process_sample(df.iloc[i], dir_out, encodings)
# # df.apply(lambda x: , axis=1)
# # process_se(df)
# if procese_shots:
#     dir_data = '/home/jrobby/kinship/'
#     dirs_scenes = glob.glob(dir_data + '*/scenes/')
#     dirs_scenes.sort()
#
#     subjects = [d.replace(dir_data, '').replace('/scenes/', '') for d in
#                 dirs_scenes]
#
#     for subject in subjects:
#         dout = dir_scenes + subject + '/scenes/faces/'
#         print(subject)
#         if not (subject in df.ref.to_list() and not Path(dout).exists()):
#             continue
#         mid = df.loc[df.ref == subject, 'mid'].values[0]
#         fid = df.loc[df.ref == subject, 'fid'].values[0]
#         dir_mid = f"{dir_fids}{fid}/MID{mid}/"
#         encodings = encode_mids(dir_mid)
#         encodings = list(encodings.values())
#         process_scenes(dir_scenes + subject + '/scenes/', dout, encodings)
