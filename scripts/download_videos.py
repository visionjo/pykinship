import pandas as pd
import wget
from pathlib import Path
from pytube import YouTube

from src.data.videos import read_family_member_list

f_urls = "../data/family_members.csv"
d_out = "../data/fiw-videos/new-raw/"
df = read_family_member_list(f_urls)

cols = df.columns.to_list()
video_cols = [c for c in cols if c.count('video')]
df_videos = df[['ref'] + ['fid'] + ['mid'] + video_cols]


def process_subject(row, dir_out=d_out):
    ref = row['ref']
    fid = f"{row['fid']}.{row['mid']}"

    dout = Path(dir_out).joinpath(ref)
    with open(dout.joinpath('fiw.txt'), 'w') as f:
        f.write(fid)

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
