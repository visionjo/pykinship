from pathlib import Path

import pandas as pd
import scipy.io.wavfile as wav


def cut(data, freq, start, end):
    """
    Cut track array from start (in seconds) to end (in seconds)
    or till end of track if end second is bigger then track length
    :param track: wav audio data
    :param start: start (in seconds)
    :param end: end (in seconds)
    :param freq: frequency of audio data
    :return:
    """
    end = int(end * freq)
    if end > len(data):
        return data[int(start * freq) :]
    return data[int(start * freq) : end]


def to_mono(fname, channel=0):
    """
    Opens wav file and returns it as mono file if stereo
    :param fname: file name
    :param channel: channel index - default 0
    :return: tuple of frequency and data numpy array
    """
    (freq, sig) = wav.read(fname)
    if sig.ndim == 2:
        return (sig[:, channel], freq)
    return (sig, freq)


def time2sec(str_in):
    time_sec = float(str_in.split(":")[0]) * 60  # convert minutes
    time_sec += float(str_in.split(":")[1].split(".")[0])
    time_sec += float(str_in.split(".")[1]) * 10 ** (-float(len(str_in.split(".")[1])))
    return time_sec


def seconds(data, freq):
    """
    Returns number of seconds from track data based on frequency
    :param track: wav audio data
    :param freq: frequency of audio data
    :return: number of seconds
    """
    return len(data) / freq


def save(filename, data, freq):
    """
    Wrapper for scipy.io.wavfile write method
    :param filename: name of wav file
    :param freq: frequency of audio data
    :param data: wav audio data
    """
    wav.write(filename=filename, rate=freq, data=data)


# Set chunk size of 1024 samples per data frame
chunk = 1024

path_in = Path("/Volumes/MyWorld/FIW-MM/speaker-diarization")
path_out = path_in / "clips"
path_out.mkdir(exist_ok=True, parents=True)

f_diarization = list(path_in.glob("v*.csv"))

path_audio = Path("/Volumes/MyWorld/FIW-MM/raw/audio/")

for f_csv in f_diarization:
    df_diarization = pd.read_csv(f_csv)
    dout = path_out / f_csv.name.split("-")[0]
    try:
        dout.mkdir(parents=True)
    except Exception:
        print("skipping", f_csv)
        continue
    speaker_ids = df_diarization.speaker.unique()
    # Open the sound file
    fin = path_audio / f_csv.with_suffix(".wav").name.replace("-diarization", "")
    signal, freq = to_mono(fin)

    for speaker_id in speaker_ids:
        df_cur = df_diarization.loc[df_diarization.speaker == speaker_id]
        #
        dir_out = dout / f"s{speaker_id}"
        dir_out.mkdir(exist_ok=True, parents=True)
        counter = 1
        for start, end in zip(df_cur.start.to_list(), df_cur.stop.to_list()):
            stime = time2sec(start)
            etime = time2sec(end)
            audio_snippet = cut(signal, freq, stime, etime)

            save(dir_out / f"u{counter}.wav", audio_snippet, freq)
            counter += 1
