import tqdm
import wget
from pathlib import Path
from pytube import YouTube

from src.data.videos import read_family_member_list

f_urls = "../data/family_members.csv"
# f_video = "../data/videos-fiw/Ben_Affleck.mp4"
# f_encodings = f"{str(Path(f_video).parent)}/mid-encodings.pkl"
# dir_out = "/Volumes/MySpace/kinship/"
# dir_scenes = '/home/jrobby/kinship/'
# dir_fids = '/home/jrobby/master-version/fiwdb/FIDs/'  # '/Users/jrobby/data/FIDs/'
df = read_family_member_list(f_urls)

# YouTube('https://youtu.be/9bZkp7q19f0').streams.get_highest_resolution().download()
#
li_videos = list(zip(df.ref.values, df.video.values)) + list(zip(df.ref.values, df.video2.values))
li_videos = [l for l in li_videos if str(l[1]) != 'nan']
obin = 'raw/'
for ref, url in tqdm.tqdm(reversed(li_videos[300:])):
    dout = obin + ref + '/video/'
    if Path(dout).is_dir():
        dout = dout.replace('video', 'video2')
    Path(dout).mkdir(parents=True, exist_ok=True)

    yt = YouTube(url)

    url_t = yt.thumbnail_url
    t = wget.download(url_t, out=dout + 'thumbnail.png')
    describe = yt.description
    title = yt.title
    length = yt.length
    with open(dout + 'meta.txt', 'w') as f:
        f.write(url + '/n/n')
        f.write(title + '/n')

        f.write("length: " + str(length))
        f.write("thumbnail url:" + url_t + '/n')
        f.write(describe + '/n/n')
    caption = yt.captions.get_by_language_code('en')

    if caption:
        cout = caption.generate_srt_captions()
        with open(dout + 'caption.txt', 'w') as f:
            f.write(cout)

    stream = yt.streams.first()
    stream.download(dout)
