#!/usr/bin/env python
# coding: utf-8

# # Demo for RFIW-2020 Task 1 (Phase 1).
# ## Kinship Verification
#
# This is a basic demo showing some tricks for using pandas for verification
# task evaluation, along with analysis.
#
# Note that it is assumed features are extracted in are stored with the same
# name a image files, except as PKL files. The demo loads all features into a
# dictionary with keys set as the image (face) name and path
# (i.e., FID/MID/faceID), but with the extension omitted. Thus, modifications
# can easily be made in data loading cell to fit the scheme in place if
# different.
#
# For this, faces were encoded using SphereFace trained on MSCeleb in Pytorch
# (though any features can be plugged in).
#
# No fine-tuning or special tricks were employed. This is solely to demonstrate
# a few simple steps for evaluation, followed
# by easy to generate, yet appealing and insightful, visualizations of the
# feature embeddings.

import time

import matplotlib.pyplot as plt
import numpy as np
import swifter
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

print(swifter.__version__)

# set styles for figures
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style(
    "whitegrid",
    {"font.family": "serif", "font.serif": "Times New Roman", "fontsize": 18},
)

# here set the baths to pairs list, along with directory containing features
dir_root = "../data/rfiw2020/"
dir_task = f"{dir_root}/verification/"
dir_test = dir_task + "test/"
dir_features = f"{dir_root}/FIDs-features/"

f_ground_truth = f"{dir_test}ref.csv"
df_true_labels = pd.read_csv(f_ground_truth)

# load pairs as DataFrame (from PKL file)
df_pairlist = pd.read_pickle(dir_test + "ref.pkl")
# set tags to uppercase for formatting later
df_pairlist.ptype = df_pairlist.ptype.str.upper()
# get all unique relationship types in DF
relationship_types = df_pairlist.ptype.unique()

print(
    "Processing {} pairs of {} relationship types".format(
        len(df_pairlist), len(relationship_types)
    )
)
print(relationship_types)

li_images = list(np.unique(df_pairlist.p1.to_list() + df_pairlist.p2.to_list()))

# load all features in LUT (ie dictionary)
# f_features = [dir_features + f.replace('.jpg', '.pkl') for f in li_images]
features = {
    f: pd.read_pickle(dir_features + f.replace(".jpg", ".pkl")) for f in li_images
}

start = time.time()
# score all pairs, because L2 norm applied on features dot is same as cosine sim
df_pairlist["score"] = df_pairlist.swifter.apply(
    lambda x: np.dot(features[x.p1], features[x.p2].T), axis=1
)
t2 = time.time() - start
print(t2)

print(df_pairlist.head())

print(df_pairlist.tail())

df_pairlist["label"] = df_pairlist["labels"]
# 'tags' column in for formatting legend in violin-plot in couple of cells below
df_pairlist["tags"] = "KIN"
df_pairlist.loc[df_pairlist.label == 0, "tags"] = "NON-KIN"

fpr, tpr, threshold = roc_curve(df_pairlist.label.values, df_pairlist.score.values)
auc = roc_auc_score(df_pairlist.label.values, df_pairlist.score.values)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC (AUC: %0.2f)" % auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC: 0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC All Types Averaged")
plt.legend(loc="best")
plt.show()

# Next plot signal detection models (SDM) for each relationship type. From this,
# the distribution of scores as a function of label (i.e., KIN vs NON-KIN)
# can be compared.
df_pairlist.label = df_pairlist.label.astype(int)

sns.despine(left=True)
f, axs = plt.subplots(
    2,
    int(np.ceil(len(relationship_types) / 2)),
    figsize=(15, 7),
    sharex=True,
    sharey=True,
)
i = [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 0],
    [0, 5],
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
]
for j, att in enumerate(relationship_types):
    df_cur = df_pairlist.loc[df_pairlist.ptype == att, ["score", "label"]]
    sns.distplot(
        df_cur.loc[df_cur.label == 1, "score"],
        hist=True,
        label="True",
        ax=axs[i[j][0], i[j][1]],
        color="g",
    )
    sns.distplot(
        df_cur.loc[df_cur.label == 0, "score"],
        hist=True,
        label="False",
        ax=axs[i[j][0], i[j][1]],
        color="r",
    )
    axs[i[j][0], i[j][1]].set_title(att)
    axs[i[j][0], i[j][1]].set_xlabel("")
plt.show()
# Similar to SDM, but let's look at boxen plots as means of another
# visualization of two-class separability.

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
sns.despine(left=True)

sns.violinplot(
    x="ptype",
    y="score",
    data=df_pairlist,
    hue="tags",
    ax=ax,
    linewidth=2.5,
    width=0.75,
    palette="Pastel1",
    order=["BB", "SS", "SIBS", "FD", "FS", "MD", "MS", "GFGD", "GFGS", "GMGD", "GMGS"],
)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Relationship Type")

ax.set_ylabel("Similarity Score")
# Calculate number of obs per group & median to position labels
medians = df_pairlist.groupby(["ptype"])["score"].min().values
nobs = df_pairlist["ptype"].value_counts().values
nobs = [str(f"{x:,}") for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

# Add it to the plot
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
    value = pos[tick]
    ax.text(
        value,
        -0.45,
        nobs[tick],
        horizontalalignment="center",
        size="small",
        color="k",
        weight="semibold",
    )

plt.legend(loc="best")

plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.gca()
sns.despine(left=True)
df_pairs = df_pairlist.loc[not df_pairlist.ptype.str.contains("G")]
sns.violinplot(
    x="ptype",
    y="score",
    data=df_pairs,
    hue_order=["NON-KIN", "KIN"],
    split=True,
    hue="tags",
    ax=ax,
    linewidth=2.5,
    width=0.75,
    palette="Pastel1",
    order=["BB", "SS", "SIBS", "FD", "FS", "MD", "MS"],
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Relationship Type")

ax.set_ylabel("Similarity Score")
# Calculate number of obs per group & median to position labels
medians = df_pairs.groupby(["ptype"])["score"].min().values
nobs = df_pairs["ptype"].value_counts().values
nobs = [str(f"{x:,}") for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

# Add it to the plot
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
    value = pos[tick]
    ax.text(
        value,
        -0.4,
        nobs[tick],
        horizontalalignment="center",
        size="small",
        color="k",
        weight="semibold",
    )

plt.legend(loc="best")

plt.show()
