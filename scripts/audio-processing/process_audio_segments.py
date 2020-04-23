# import torch
from pathlib import Path
#
# path_audio = Path('/Users/jrobby/GitHub/pykinship/data/fiw-mm/FIDs-MM/audio/wav/')
# emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
#
# for path in path_audio.rglob('*.wav'):
#     print(path)
#     # embeddings = emb(test_file)
# load model
import torch
from tqdm import tqdm

model = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')

print(f'Embedding has dimension {model.dimension:d}.')
# Embedding has dimension 512.

# extract speaker embedding on the whole file using built-in sliding window
import numpy as np
from pyannote.core import Segment

path_audio = Path('/Users/jrobby/GitHub/pykinship/data/fiw-mm/FIDs-MM/audio/wav/')
path_out = Path('/Volumes/MyWorld/FIW-MM/features/audio/librosaMFCCami')
path_out.mkdir(exist_ok=True, parents=True)
paths_all = list(path_audio.rglob('*.wav'))
# emb_voxceleb
for path in tqdm(paths_all):
    fout = Path(str(path).replace(str(path_audio), str(path_out)).replace('.wav', '.npy'))
    if fout.is_file():
        continue
    print(path)
    embedding = model({'audio': str(path)})

    embedding = np.array([em[1] for em in embedding])

    fout.parent.mkdir(exist_ok=True, parents=True)
    # embeddings = emb(test_file)
    np.save(fout, embedding)
alias
sync_video = 'rsync -auvzP --exclude "*.mp4" --exclude "*.jpg" /Volumes/MyWorld/FIW-MM/features/video /Users/jrobby/Dropbox/FIW_Video/data/FIDs-MM-features/visual/'

# A  # for window, emb in embedding:
#     assert isinstance(window, Segment)
#     assert isinstance(emb, np.ndarray)

# # extract speaker embedding of an excerpt
# excerpt1 = Segment(start=2.3, end=4.8)
# emb1 = model.crop({'audio': '/path/to/your/audio.wav'}, excerpt1)
# assert isinstance(emb1, np.ndarray)
#
# # compare speaker embedding
# from scipy.spatial.distance import cdist
#
# excerpt2 = Segment(start=5.2, end=8.3)
# emb2 = model.crop({'audio': '/path/to/your/audio.wav'}, excerpt2)
# distance = cdist(np.mean(emb1, axis=0, keepdims=True),
#                  np.mean(emb2, axis=0, keepdims=True),
#                  metric='cosine')[0, 0]
