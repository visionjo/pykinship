import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

#############################################################################
# Plots ROC curve
#############################################################################
sns.set()

warnings.filterwarnings("ignore")


def load_actives(fname):
    actives = []
    for line in open(fname, "r").readlines():
        id = line.strip()
        actives.append(id)

    return actives


# def load_data(fname, ):
#
#     for f in dir_results.glob('*fusion*.csv'):
#         print(f)
#
#
#     sfile = open(fname, 'r')
#     label = sfile.readline()
#     label = label.strip()
#
#     scores = []
#     for line in sfile.readlines():
#         id, score = line.strip().split()
#         scores.append((id, float(score)))
#
#     return label, scores


def load_dataframes(din, wildcard="*fusion*.csv"):
    data = {}
    for f in din.glob(wildcard):
        print(f)
        ref = "-".join(f.with_name("").name.split("_")[-3:]).replace("-fusion", "")
        print(ref)
        data[ref] = pd.read_csv(f)

    # sfile = open(fname, 'r')
    # label = sfile.readline()
    # label = label.strip()
    #
    # scores = []
    # for line in sfile.readlines():
    #     id, score = line.strip().split()
    #     scores.append((id, float(score)))
    #
    # return label, scores


def get_rates(actives, scores):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :rtype: tuple(list[float], list[float])
    """

    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    nractives = len(actives)
    nrdecoys = len(scores) - len(actives)

    foundactives = 0.0
    founddecoys = 0.0
    for idx, (id, score) in enumerate(scores):
        if id in actives:
            foundactives += 1.0
        else:
            founddecoys += 1.0

        tpr.append(foundactives / float(nractives))
        fpr.append(founddecoys / float(nrdecoys))

    return tpr, fpr


def setup_roc_curve_plot(plt):
    """
    :type plt: matplotlib.pyplot
    """

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)


def save_roc_curve_plot(plt, filename, randomline=False):
    """
    :type plt: matplotlib.pyplot
    :type fname: string
    :type randomline: boolean
    """

    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle="dashed", color="black", linewidth=2, label="random")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    # plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(filename, transparent=True)


def depict_roc_curve(actives, scores, label, color, filename, randomline=True):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type fname: string
    :type randomline: boolean
    """

    plt.figure(figsize=(4, 4), dpi=80)

    setup_roc_curve_plot(plt)
    add_roc_curve(plt, actives, scores, color, label)
    save_roc_curve_plot(plt, filename, randomline)


def add_roc_curve(plt, actives, scores, color, label):
    """
    :type plt: matplotlib.pyplot
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type label: string
    """

    tpr, fpr = get_rates(actives, scores)
    roc_auc = auc(fpr, tpr)

    roc_label = "{} (AUC={:.3f})".format(label, roc_auc)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=roc_label)


def is_supported_image_type(ext):
    fig = plt.figure()
    return ext[1:] in fig.canvas.get_supported_filetypes()


def load_scores(path_in: Path) -> dict:
    f_scores = list(path_in.glob("*.csv"))
    # f_scores = [f for f in f_scores if not "min" in str(f)]
    return {
        f.name: pd.read_csv(
            f,
            usecols=["label", "score", "tag"],
            dtype={"label": np.int, "score": np.float, "tag": np.str},
        )
        for f in f_scores
    }


def set_plot(plt, fontsize=15) -> None:
    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=fontsize)
    plt.yticks([])

    # plt.yticks(np.arange(0.0, 1.1, step=0.1))
    # plt.ylabel("True Positive Rate", fontsize=fontsize)

    # plt.title("ROC Curve Analysis", fontweight="bold", fontsize=15)
    # plt.legend(prop={"size": 10}, loc="lower right")


def calculate_roc_points(df_in: dict) -> pd.DataFrame:
    result_table = pd.DataFrame(data=None, columns=["method", "fpr", "tpr", "auc"])

    for method, df in df_in.items():
        fpr, tpr, thresh = roc_curve(df["label"], df["score"])
        auc_score = auc(fpr, tpr)
        df_tmp = pd.DataFrame([method, fpr, tpr, auc_score]).T
        df_tmp.columns = ["method", "fpr", "tpr", "auc"]
        result_table = pd.concat([result_table, df_tmp.copy()])

    result_table.reset_index(inplace=True)
    return result_table


def filter_datatables(df, keep):
    for k, v in df.items():
        df[k] = v.loc[v.tag == keep]
    return df


if __name__ == "__main__":
    dir_results = Path("../../results/trisubject_evaluation/roc")
    do_type = "FM-S"
    path_out = Path(
        "/Users/jrobby/Dropbox/FIW_Video/results/trisubject_evaluation/roc/roc-{}.pdf".format(
            do_type
        )
    )

    df_list = load_dataframes(dir_results)
    df_list = filter_datatables(df_list, do_type)
    results_summary = calculate_roc_points(df_list)
    # Define a result table as a DataFrame

    fig = plt.figure(figsize=(8, 6))

    for i in results_summary.index:
        if results_summary.loc[i]["auc"] > 0.83:
            plt.plot(
                results_summary.loc[i]["fpr"],
                results_summary.loc[i]["tpr"],
                label="{} {}, AUC={:.3f}".format(
                    results_summary.loc[i]["method"].split("_")[-3],
                    results_summary.loc[i]["method"].split("_")[-1].replace(".csv", ""),
                    results_summary.loc[i]["auc"],
                ),
            )
    set_plot(plt)
    # plt.show()
    save_roc_curve_plot(plt, path_out)
