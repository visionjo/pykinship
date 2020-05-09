from warnings import warn

import pandas as pd
import swifter
import wget
from pathlib import Path
from pytube import YouTube

# from swifter import swifter as sw
print(str(swifter.__version__))


def process_subject(row, dir_out):
    """
    Download videos per subject
    :param
    """
    ref = row["ref"]
    fid = f"{row['fid']}.{row['mid']}"

    dout = Path(dir_out).joinpath(ref)
    with open(dout.joinpath("fiw.txt"), "w") as f:
        f.write(fid)

    vlist = [
        (k, v)
        for k, v in row.items()
        if pd.notna(v)
        if (k != "ref") and (k != "fid") and (k != "mid")
    ]
    for v in vlist:
        if Path(dout).joinpath(v[0]).is_dir():
            pass
            # print(dout.joinpath(v[0]))
        else:
            print(dout, v)
            do = str(Path(dout).joinpath(v[0])) + "/"
            url = v[1]

            Path(do).mkdir(parents=True, exist_ok=True)
            yt = YouTube(url)

            url_t = yt.thumbnail_url
            _ = wget.download(url_t, out=do + "thumbnail.png")
            describe = yt.description
            title = yt.title
            length = yt.length

            with open(do + "meta.txt", "w") as f:
                f.write(url + "/n/n")
                f.write(title + "/n")

                f.write("length: " + str(length))
                f.write("thumbnail url:" + url_t + "/n")
                f.write(describe + "/n/n")
            caption = yt.captions.get_by_language_code("en")

            if caption:
                cout = caption.generate_srt_captions()
                with open(do + "caption.txt", "w") as f:
                    f.write(cout)

            stream = yt.streams.first()
            stream.download(do)


def fetch_video(row, dir_out):
    """
    Download videos per VID
    :param
    """
    vid, fid, url = row.values
    # output pointers
    dout = str(Path(dir_out).joinpath(fid).joinpath(vid)) + "/"

    print(dout, vid)
    try:
        Path(dout).mkdir(parents=True)
    except Exception as e:
        print(dout, "already exists", e.message)
        return
    try:
        # youtube handler
        yt = YouTube(url)
    except Exception as e:
        warn("Unable to load url", row, e.message)
        return
    # download thumbnail
    url_t = yt.thumbnail_url
    try:
        _ = wget.download(url_t, out=dout + "thumbnail.png")
    except Exception as e:
        warn(vid, "no thumbnail downloaded", e.message)

    # prepare metadata to dump to file
    with open(dout + "meta.txt", "w") as f:
        f.write(url + "/n/n")
        f.write(yt.title + "/n")
        f.write("length: " + str(yt.length))
        f.write("thumbnail url:" + url_t + "/n")
        f.write(yt.description + "/n/n")

    try:
        caption = yt.captions.get_by_language_code("en")
        if caption:
            cout = caption.generate_srt_captions()
            with open(dout + "caption.txt", "w") as f:
                f.write(cout)
    except Exception as e:
        warn("Unable to download captions", dout, e.message)

    stream = yt.streams.first()
    stream.download(dout)


f_master = "../data/fiw-videos/meta/fiw-vid-master.csv"
d_out = "../data/fiw-videos/new-raw/"
df = pd.read_csv(f_master)

df_list = df[["vid", "fid", "url"]].drop_duplicates()

_ = [Path(d_out + d).mkdir(parents=True, exist_ok=True) for d in df_list.fid.unique()]

df_list.swifter.apply(lambda x: fetch_video(x, d_out), axis=1)
