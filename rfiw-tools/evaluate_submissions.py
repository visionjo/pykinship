import glob

import pandas as pd
from tqdm import tqdm

dir_submissions = "../data/rfiw2020/submissions-t1/"

f_ground_truth = "../data/rfiw2020/verification/test/ref.csv"

f_predictions = glob.glob(f"{dir_submissions}*/*.csv")

print(len(f_predictions))

df_true_labels = pd.read_csv(f_ground_truth)
df_true_labels["label"] = df_true_labels["labels"]

del df_true_labels["labels"]

df_table = df_true_labels.groupby("ptype").sum()["label"]
df_table = df_table.append(pd.Series([0], index=["avg"]))

df_table = pd.DataFrame(df_table)
del df_table[0]

n_pairs = len(df_true_labels) * 1.0
for f_prediction in tqdm(f_predictions):
    # get submission identifiers
    user, submission_id = f_prediction.split("/")[-2:]
    submission_id = submission_id.replace(".csv", "")

    # prepare submission
    df_predictions = pd.read_csv(f_prediction)
    df_predictions["label"] = df_predictions["label"].astype(int)

    # evaluate using ground-truth
    df_true_labels["prediction"] = df_predictions["label"]

    df_true_labels["correct"] = df_true_labels["prediction"] == df_true_labels["label"]

    df_group_by_type = df_true_labels.groupby("ptype")
    acc_per_category = (
        df_group_by_type.sum()["correct"] / df_group_by_type.count()["correct"]
    )

    acc = df_true_labels["correct"].sum() / n_pairs
    s_acc = pd.Series([acc], index=["avg"])

    acc_per_category = acc_per_category.append(s_acc)

    df_table[user] = acc_per_category

df_table.to_csv("table-summary2.csv")
