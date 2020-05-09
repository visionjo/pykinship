from pathlib import Path
import pandas as pd
from tqdm import tqdm
from imutils.paths import list_files
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# compare embeddings
if __name__ == "__main__":
    do_mids = False
    encode_faces = False
    create_montages = True
    dir_data = "/Users/jrobby/data/nature/features/"

    dirs_fid = Path(dir_data).glob("F????")

    for dir_fid in tqdm(dirs_fid):
        dirs_mid = dir_fid.glob("MID*")
        if create_montages:
            paths = [l for l in list_files(dir_fid) if l.count("msceleb")]
            paths.sort()

            nmembers = len(set([Path(p).parent.parent.as_posix() for p in paths]))
            encodings = [pd.read_pickle(path) for path in paths]
            encodings = [e for e in encodings if not type(e) is int]

            cs_matrix = cosine_similarity(encodings)
            sns.heatmap(cs_matrix)
            plt.tight_layout()
            plt.savefig(dir_fid.as_posix() + "_heatmaps.pdf")
            plt.close("all")
