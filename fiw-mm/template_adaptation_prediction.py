"""
Use templates from template_adaptation.py to predict for the verification task.

Usage: template_adaptation.py <verification-lists> <templates> [--normalize]

Notes:
<verification-lists> is the path to the test, train, and val splits of Track I.
<templates> is a directory where the output of template_adaptation.py was placed.
"""
from pathlib import Path
import pickle
from tqdm.auto import tqdm
from docopt import docopt
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, accuracy_score


if __name__ == "__main__":
    pd.set_option("display.max_columns", 10)
    args = docopt(__doc__)
    lists = Path(args["<verification-lists>"])
    templates = Path(args["<templates>"])
    should_normalize = args["--normalize"]

    test = pd.read_csv(lists / "test.csv").fillna("NON-KIN")
    with open(templates / "svm_bank.pkl", "rb") as f:
        svm_bank = pickle.load(f)

    with open(templates / "mid_feature_bank.pkl", "rb") as f:
        feature_bank = pickle.load(f)

    scores = []
    output_dataframe = pd.DataFrame()
    for row in tqdm(test.itertuples()):   
        p1_svm = svm_bank[row.p1]
        p2_svm = svm_bank[row.p2]

        p1_template = np.mean(feature_bank[row.p1], axis=0).reshape(1, -1)
        p2_template = np.mean(feature_bank[row.p2], axis=0).reshape(1, -1)

        if should_normalize:
            p1_template = normalize(p1_template)
            p2_template = normalize(p2_template)

        score_1 = p1_svm.decision_function(p2_template)
        score_2 = p2_svm.decision_function(p1_template)
        scores.append(np.mean([score_1, score_2]))

    test["score"] = pd.Series(scores)


    ptypes_to_include = {"F-S", "F-D", "M-S", "M-D", "B-B", "S-S", "SIBS", "NON-KIN"}
    test = test[test.ptype.isin(ptypes_to_include)]
    fpr, tpr, thresholds = roc_curve(test.label, test.score)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    test["prediction"] = test.score > best_threshold


    for ptype, pairs_of_ptype in test.groupby("ptype"):
        score_for_ptype = accuracy_score(
            pairs_of_ptype.label, pairs_of_ptype.prediction
        )
        output_dataframe.loc["template_adaptation", ptype] = score_for_ptype
    output_dataframe.loc["template_adaptation", "all"] = accuracy_score(
        test.label, test.prediction
    )
    print(output_dataframe)