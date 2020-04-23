#!/usr/bin/env python

from glob import glob
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import tqdm

from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import mfcc

verbose = 0
path_data = Path("../../data/fiw-mm/FIDs-MM/")
path_out = Path("../../data/fiw-mm/features/audio/")
dlist = glob(str(path_data) + "/F????/MID*/clips/audio")

for path_clips in tqdm.tqdm(dlist):
    print(path_clips)

    paths_wavs = glob(path_clips + "/*.wav")
    for path_wav in paths_wavs:
        if verbose:
            print(path_wav)
        (rate, sig) = wav.read(path_wav)
        mfcc_feat = mfcc(sig, rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(sig, rate)

        fout = Path(
            path_wav.replace(str(path_data), str(path_out))
            .replace("clips/", "")
            .replace(".wav", ".npy")
        )
        fout.parent.mkdir(parents=True, exist_ok=True)
        np.save(fout, fbank_feat)
