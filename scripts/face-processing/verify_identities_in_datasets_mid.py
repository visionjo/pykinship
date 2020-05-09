import pickle

import numpy as np
import shutil
from html4vision import Col, imagetable
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def to_html(imfiles_keep, imfiles_go, mid):
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
        dout + f"{mid}-face-cleanup.html",
        "Sample Decision and Score",
        summary_row=summary_row,  # add a summary row showing overall statistics of the dataset
        summary_color="#fff9b7",  # highlight the summary row
        sortable=True,  # enable interactive sorting
        sticky_header=True,  # keep the header on the top
        style="img {border: 1px solid black;};",
        sort_style="materialize",  # use the theme "materialize" from jquery.tablesorter
        zebra=True,  # use zebra-striped table
    )


path_fids = Path("../../data/fiw-videos/FIW-MM/")
fn_features = "encodings.pkl"
# def process_mid():
ii = 0
for path_fid in tqdm(path_fids.glob("F????")):
    if ii < 422:
        ii += 1
        continue
    else:
        print(path_fid)
    for path_mid in path_fid.glob("MID*"):
        imfile = list(path_mid.glob("n*"))
        imfile.sort()
        if len(imfile):
            imfile2 = [
                p
                for p in path_mid.glob("*.jpg")
                if not str(p).split("/")[-1][0].startswith("n")
            ]
            imfile2.sort()
            fin = path_mid.joinpath("encodings").joinpath(fn_features)
            with open(fin, "rb") as f:
                encodings = pickle.load(f)
            keys_mid = [
                str(f).replace(path_fids._str, "").replace("/F", "F") for f in imfile2
            ]
            keys_query = [
                str(f).replace(path_fids._str, "").replace("/F", "F") for f in imfile
            ]
            if not len(keys_mid) or not len(keys_query):
                continue
            encodings_mid = [encodings[k].numpy()[1] for k in keys_mid]
            encodings_query_tup = []
            for k in keys_query:
                if k in encodings:
                    if len(encodings[k].numpy()) > 1:
                        encodings_query_tup.append((k, encodings[k].numpy()[1]))
                    else:
                        encodings_query_tup.append(
                            (k, (encodings[k].numpy().squeeze()))
                        )
            # encodings_query_tup = [(k, encodings[k].numpy()[1]) for k in keys_query if k in encodings]
            if not len(encodings_query_tup) or not len(encodings_mid):
                continue

            scores_matrix = cosine_similarity(
                np.array([e[1] for e in encodings_query_tup]),
                np.array([ee for ee in encodings_mid]),
            )
            # scores_matrix = cosine_similarity([e[1] for e in encodings_query_tup], encodings_mid)
            scores = np.median(scores_matrix, axis=1)
            mask = scores > 0.2

            score_tup = np.array(
                [(f, sc) for f, sc in zip([e[0] for e in encodings_query_tup], scores)]
            )
            imkeep, imdrop = [], []
            impath = None
            for f_tup, keep in zip(score_tup, mask):
                impath = path_fids.joinpath(
                    "/".join(f_tup[0].split("/")[:2]) + "/" + f_tup[0].split("/")[-1]
                )

                if keep:
                    imkeep.append((str(impath), f_tup[1]))
                else:
                    imdrop.append((str(impath), f_tup[1]))
            mid = str(impath.parent).split("/")[-1].lower()
            print(impath)
            to_html(imkeep, imdrop, mid)
            for f_im in imkeep:
                # path_out = Path(f_im[0] + 'faces-cleaned')
                # if not path_out.is_file():
                Path(f_im[0]).parent.joinpath("faces-cleaned").mkdir(exist_ok=True)
                shutil.copyfile(
                    f_im[0],
                    Path(f_im[0])
                    .parent.joinpath("faces-cleaned")
                    .joinpath(f_im[0].split("/")[-1]),
                )
