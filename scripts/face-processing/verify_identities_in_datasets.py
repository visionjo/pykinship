import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import shutil

from html4vision import Col, imagetable, imagetile


def to_html(imfiles_keep, imfiles_go):
    scores_keep = [str(sc) for _, sc in imfiles_keep]
    scores_go = [str(sc) for _, sc in imfiles_go]

    finfo = imfiles_keep[0][0] if len(imfiles_keep) else imfiles_go[0][0]
    dout = finfo.replace("/".join(finfo.split("/")[-2:]), "")

    cols = [
        Col("id1", "ID"),  # 1-based indexing
        Col(
            "img",
            "keep",
            [f.replace(dout, "") for f, _ in imfiles_keep],
            None,
            "background: #28aade",
            href=[f.replace(dout, "") for f, _ in imfiles_keep],
        ),
        Col("text", "keep (score)", scores_keep),
        Col(
            "img",
            "dump",
            [f.replace(dout, "") for f, _ in imfiles_go],
            None,
            "background: #db8a0f",
            href=[f.replace(dout, "") for f, _ in imfiles_go],
        ),
        Col("text", "dump (score)", scores_go, style="text {width: 2in;}"),
    ]
    summary_row = [
        "ID",
        len(imfiles_keep),
        np.mean([float(sc) for _, sc in imfiles_keep]),
        len(imfiles_go),
        np.mean([float(sc) for _, sc in imfiles_go]),
    ]

    imagetable(
        cols,
        dout + "face-cleanup.html",
        "Sample Decision and Score",
        summary_row=summary_row,  # add a summary row showing overall statistics of the dataset
        summary_color="#fff9b7",  # highlight the summary row
        sortable=True,  # enable interactive sorting
        sticky_header=True,  # keep the header on the top
        style="img {border: 1px solid black;};",
        sort_style="materialize",  # use the theme "materialize" from jquery.tablesorter
        zebra=True,  # use zebra-striped table
    )


path_fids = Path("../../data/fiw-videos/FIDs/")
fn_features = "encodings.pkl"
# def process_mid():
for path_fid in tqdm(path_fids.glob("F????")):
    for path_mid in path_fid.glob("MID*"):
        imfile = list(path_mid.joinpath("faces").glob("n*"))
        imfile.sort()
        if len(imfile):
            imfile2 = [
                p
                for p in path_mid.joinpath("faces").glob("*.jpg")
                if not str(p).split("/")[-1][0].startswith("n")
            ]
            imfile2.sort()
            fin = path_mid.joinpath(fn_features)
            with open(fin, "rb") as f:
                encodings = pickle.load(f)
            keys_mid = [
                str(f)
                .replace(path_fids._str, "")
                .replace("faces/", "")
                .replace(".jpg", "")
                .replace("/F", "F")
                for f in imfile2
            ]
            keys_query = [
                str(f)
                .replace(path_fids._str, "")
                .replace("faces/", "")
                .replace(".jpg", "")
                .replace("/F", "F")
                for f in imfile
            ]

            encodings_mid = [encodings[k] for k in keys_mid]
            encodings_query_tup = [
                (k, encodings[k]) for k in keys_query if k in encodings
            ]
            if not len(encodings_query_tup) or not len(encodings_mid):
                continue

            cosine_similarity([e[1] for e in encodings_query_tup], encodings_mid)
            scores_matrix = cosine_similarity(
                [e[1] for e in encodings_query_tup], encodings_mid
            )
            scores = np.median(scores_matrix, axis=1)
            mask = scores > 0.2

            score_tup = np.array(
                [(f, sc) for f, sc in zip([e[0] for e in encodings_query_tup], scores)]
            )
            imkeep, imdrop = [], []
            for f_tup, keep in zip(score_tup, mask):
                impath = path_fids.joinpath(
                    "/".join(f_tup[0].split("/")[:2])
                    + "/faces/"
                    + f_tup[0].split("/")[-1]
                    + ".jpg"
                )
                if keep:
                    imkeep.append((str(impath), f_tup[1]))
                else:
                    imdrop.append((str(impath), f_tup[1]))
            to_html(imkeep, imdrop)
            for f_im in imkeep:
                path_out = Path(f_im[0].replace("faces", "faces-cleaned"))
                if not path_out.is_file():
                    Path(f_im[0]).parent.parent.joinpath("faces-cleaned").mkdir(
                        exist_ok=True
                    )
                    shutil.copyfile(f_im[0], path_out)
