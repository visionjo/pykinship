import numpy as np
from html4vision import Col, imagetable

def create_html_page_face_montage(p1, p2, labels):

    neg_ids = np.where(labels == 0)[0][-100:]
    pos_ids = np.where(labels == 1)[0][:100]

    p1_pos = p1[pos_ids]
    p1_neg = p1[neg_ids]
    p2_pos = p2[pos_ids]
    p2_neg = p2[neg_ids]

    hard_pos_pairs = np.array(list(zip(p1_pos, p2_pos)))
    hard_neg_pairs = np.array(list(zip(p1_neg, p2_neg)))

    cols = [Col('text', 'FID1', t1[pos_ids]),
            Col('img', 'P1', [CONFIGS.path.dfid + f + '.jpg' for f in list(hard_pos_pairs[:, 0])]),
            Col('img', 'P2', [CONFIGS.path.dfid + f + '.jpg' for f in list(hard_pos_pairs[:, 1])]),
            Col('text', 'Scores', ["{0:0.5}".format(sc * 100) for sc in scores[pos_ids]])
            ]
    # cols2 = []
    imagetable(cols,
               imscale=0.75,  # scale all images to 50%
               sticky_header=True,  # keep the header on the top
               out_file=dir_fold + 'hard_positives.html',
               style='img {border: 1px solid black;-webkit-box-shadow: 2px 2px 1px #ccc; box-shadow: 2px 2px 1px #ccc;}',
               )
    cols = [Col('text', 'FID1', ["{}\n{}".format(tt1, tt2) for (tt1, tt2) in zip(t1[neg_ids], t2[neg_ids])]),
            Col('img', 'P1', [CONFIGS.path.dfid + f + '.jpg' for f in list(hard_neg_pairs[:, 0])]),
            Col('img', 'P2', [CONFIGS.path.dfid + f + '.jpg' for f in list(hard_neg_pairs[:, 1])]),
            # Col('text', 'FID2', t2[neg_ids]),
            Col('text', 'Scores', ["{0:0.5}".format(sc * 100) for sc in scores[neg_ids]])]

    imagetable(cols,
               imscale=0.75,  # scale all images to 50%
               sticky_header=True,  # keep the header on the top
               out_file=dir_fold + 'hard_negatives.html',
               style='img {border: 1px solid black;-webkit-box-shadow: 2px 2px 1px #ccc; box-shadow: 2px 2px 1px #ccc;}',
               )
