"""
Plotting module

TODO - add argument for log-scales
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import auc, roc_curve

params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
sns.set(context="paper", style="whitegrid", font_scale=1, rc=params)
params_font = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "font.color": "darkred",
    "font.weight": "normal",
    "font.size": 16,
}


def set_defaults(
    font=params_font,
    rc=[("axes.facecolor", (0, 0, 0, 0))],
    style="white",
    gridstyle="whitegrid",
):
    sns.set(style=style, rc=rc)
    sns.set_style(gridstyle, font)


set_defaults()


def generate_roc(
    scores,
    labels,
    fpath="",
    calculate_auc=True,
    add_diag_line=False,
    color="darkorange",
    lw=2,
    label="ROC curve",
    title=None,
):
    """
    Parameters
    ----------
    scores: list    scores of the N pairs (len=N)
    labels: list    boolean labels of the N pairs (len=N)
    fpath:          file-path to save ROC; only saved if arg is passed in
    calculate_auc:  calculate AUC and display in legend of ROC
    add_diag_line:  add ROC curve for random (i.e., diagonal from (0,0) to (1,1)
    color:          color of plotted line
    lw:             Line width of plot
    label:          Legend Label
    title:          Axes title
    Returns Axes of figure:   plt.Axes()
    -------
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    if calculate_auc:
        roc_auc = auc(fpr, tpr)
        label += f"area = {roc_auc}"

    fig, ax = plt.subplots(1)

    plt.plot(fpr, tpr, color=color, lw=lw, label=label)

    if add_diag_line:
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
    if fpath is not None:
        plt.savefig(fpath)

    return ax


def plot_rocs(
    dataset_list,
    name_list,
    save_path=None,
    with_std=True,
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    **kwargs,
):
    f = plt.figure()

    for i, dataset in enumerate(dataset_list):
        roc_fprs = np.array(
            [protocol.curves["roc"]["fpr"] for protocol in dataset.protocols]
        )
        roc_tprs = np.array(
            [protocol.curves["roc"]["tpr"] for protocol in dataset.protocols]
        )
        roc_fprs_mean = np.mean(roc_fprs, axis=0)
        roc_tprs_mean = np.mean(roc_tprs, axis=0)
        roc_tprs_std = np.std(roc_tprs, axis=0)
        plt.plot(roc_fprs_mean, roc_tprs_mean, label=name_list[i], **kwargs)
        if with_std:
            plt.fill_between(
                roc_fprs_mean,
                roc_tprs_mean - roc_tprs_std,
                roc_tprs_mean + roc_tprs_std,
                alpha=0.3,
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc=4)

    if save_path is not None:
        f.savefig(save_path)

    plt.show()


def plot_cmc(
    dataset_list,
    name_list,
    ranks=10,
    save_path=None,
    with_std=True,
    # ylim=(0, 100),
    xlabel="Rank",
    ylabel="Face Identification Rate (%)",
    **kwargs,
):
    f = plt.figure()

    for i, dataset in enumerate(dataset_list):
        cmc = np.array([protocol.curves["cmc"] for protocol in dataset.protocols]) * 100

        cmc_mean = np.mean(cmc, axis=0)
        cmc_std = np.std(cmc, axis=0)

        plt.plot(np.arange(ranks) + 1, cmc_mean[:ranks], label=name_list[i], **kwargs)

        if with_std:
            plt.fill_between(
                np.arange(ranks) + 1,
                np.clip(cmc_mean[:ranks] - cmc_std[:ranks], a_min=0, a_max=100),
                np.clip(cmc_mean[:ranks] + cmc_std[:ranks], a_min=0, a_max=100),
                alpha=0.3,
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([1, ranks])
    plt.legend(loc=4)

    if save_path is not None:
        f.savefig(save_path)

    plt.show()


def box_plot(
    data,
    save_figure_path=None,
    fontsize=12,
    new_labels=("Imposter", "Genuine"),
    figsize=(13, 7),
):
    """
    Plot a violin plot of the distribution of the cosine similarity score of
    impostor pairs (different people) and genuine pair (same people) the plots
    are separated by ethnicity-gender attribute of the first person of each pair
    The final plot is saved to 'save_figure_path'
    Parameters
    ----------
    data:   pandas.DataFrame that contains column 'p1', 'p2', 'a1', 'a2',
            'score', and 'label' 'p1' and 'p2' are the pair of images. 'a1' and
            'a2' are the abbreviated attribute of 'p1' and 'p2' respectively.
            'score' is the cosine similarity score between 'p1' and 'p2',
            'label' is a binary indicating whether 'p1' and
            'p2' are the same person
    save_figure_path:   path to save the resulting violin plot. will not save is
                        the value is None
    """
    palette = {new_labels[0]: "orange", new_labels[1]: "lightblue"}
    data["Tag"] = data["label"]
    data.loc[data["label"] == 0, "Tag"] = new_labels[0]
    data.loc[data["label"] == 1, "Tag"] = new_labels[1]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(
        x="a1",
        y="score",
        hue="Tag",
        data=data,
        linewidth=1.25,
        dodge=True,
        notch=True,
        palette=palette,
        ax=ax,
    )
    plt.xlabel("Subgroup", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize)
    plt.title(
        "Score Distribution for Genuine and Imposter Pairs Across Subgroup",
        fontsize=fontsize,
    )

    plt.tight_layout()
    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path, transparent=True)


def draw_det_curve(
    fpr,
    fnr,
    ax=None,
    label=None,
    set_axis_log_x=True,
    set_axis_log_y=False,
    scale=100,
    title=None,
    label_x="FPR",
    label_y="FNR (%)",
    ticks_to_use_x=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0),
    ticks_to_use_y=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40),
    fontsize=24,
):
    """
    Generate DET Curve (i.e., FNR vs FPR). It is assumed FPR and FNR is
    increasing and decreasing, respectfully.
    Parameters
    ----------
    fpr: list:  false positive rate
    fnr: list:  false negative rate
    ax: plt.Axes: <default=None>:   Axes object to plot on
    label:  <default=None>
    set_axis_log_x: <default=False>
    set_axis_log_y: <default=False>
    label_x: <default='FPR',>
    label_y: <default='FNR (%)',>
    scale: <default=100>
    title: <default=None>
    ticks_to_use_x: <default=ticks_to_use_x=(1e-4, 1e-3, 1e-2, 1e-1, 1e-0)>
    ticks_to_use_y: <default=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40)>
    fontsize:  <default=24>
    Returns Axes of figure:   plt.Axes()
    -------
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(fpr, fnr * scale, label=label, linewidth=3)
    if set_axis_log_y:
        ax.set_yscale("log")
    if set_axis_log_x:
        ax.set_xscale("log")

    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(ticks_to_use_x)
    ax.set_yticks(scale * np.array(ticks_to_use_y))

    # add 10% to upper ylimit
    ax.set_ylim(0.00, scale * np.max(ticks_to_use_y))
    ax.set_xlim(np.min(ticks_to_use_x), np.max(ticks_to_use_x))
    ax.set_xlabel(label_x, fontsize=fontsize)
    ax.set_ylabel(label_y, fontsize=fontsize)

    ax.legend(loc="best")
    ax.set_title(title, fontsize=fontsize)

    return ax
