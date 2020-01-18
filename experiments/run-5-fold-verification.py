from scipy import interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# from html4vision import Col, imagetable
from src.utils import mkdir

do_scores = False
dir_data = '../data/v0.1.2/'
f_datalist = f'{dir_data}lists/verification_pairs_list_5_fold.pkl'
f_features = f'{dir_data}features/features.pkl'

dir_out = '../results/verification/baseline/'

datatable = pd.read_pickle(f_datalist)

mkdir(dir_out)
do_html=False
if do_scores or 'score' not in datatable:
    features = pd.read_pickle(f_features)

    features = {k.replace('../', ''): v for k, v in features.items()}
    datatable['score'] = datatable.apply(lambda row: np.dot(features[row['p1']], features[row['p2']]), axis=1)

# thresholds = []
ts_matches = []
sim = []
thresholds = np.arange(datatable.score.values.min(), datatable.score.values.max(), 100)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# for fold in range(CONFIGS.settings.n_folds):
folds = datatable.fold.unique()
for fold in folds:
    ids_train = datatable.fold.astype(np.int) != fold
    ids_test = datatable.fold.astype(np.int) == fold

    # print("Fold:", fold, "/", CONFIGS.settings.n_folds)
    print("Fold:", fold, "/", len(folds))

    dir_fold = f"{dir_out}fold{fold}/"
    mkdir(dir_fold)

    # # all_feats = {**tuple(trfeats1), **tuple(trfeats2)}
    # print("Normalizing Features")
    # feat_vecs = np.array(set(feats))
    # std_scale = skpreprocess.StandardScaler().fit(feat_vecs)
    # X_train = std_scale.transform(feat_vecs)
    #
    # print("PCA")
    # # pca [default k is 200]
    # X_scaled, evals, evecs = futils.pca(X_train)
    #
    # # normalize test set
    # X_feats_1 = std_scale.transform(test_feats1)
    # X_feats_2 = std_scale.transform(test_feats2)
    #
    # # apply PCA
    # X_scaled_1 = np.dot(evecs.T, X_feats_1.T).T
    # X_scaled_2 = np.dot(evecs.T, X_feats_2.T).T

    # cosine_sim = pw.cosine_similarity(X_scaled_1, X_scaled_2)
    # scores = np.diag(cosine_sim)
    scores = datatable.loc[ids_test, 'score'].values
    labels = datatable.loc[ids_test, 'label'].values.astype(np.int)

    p1 = datatable.loc[ids_test, 'p1'].values
    p2 = datatable.loc[ids_test, 'p2'].values
    t1 = datatable.loc[ids_test, 'type'].values
    # t2 = datatable.loc[ids_test, 't2'].values
    # val_labels = [int(pair[1]) for pair in pair_val_list]

    ts_matches.append(labels)
    sim.append(scores)
    # Compute ROC curve and ROC area
    # if len(thresholds) == 0:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    ids = np.argsort(scores)
    scores = scores[ids]
    labels = labels[ids]
    p1 = p1[ids]
    p1 = p1[ids]

    t1 = t1[ids]
    # t2 = t2[ids]

    neg_ids = np.where(labels == 0)[0][-100:]
    pos_ids = np.where(labels == 1)[0][:100]

    p1_pos = p1[pos_ids]
    p1_neg = p1[neg_ids]
    p2_pos = p2[pos_ids]
    p2_neg = p2[neg_ids]

    hard_pos_pairs = np.array(list(zip(p1_pos, p2_pos)))
    hard_neg_pairs = np.array(list(zip(p1_neg, p2_neg)))


    def reindex(df):
        df.index = range(len(df))

    if do_html:
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

    # else:
    #     fpr, tpr, _ = roc_curve(labels, scores, thresholds)
    roc_auc = auc(fpr, tpr)
    print("Acc:", roc_auc)
    print('Saving Results')
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    mkdir(dir_fold)
    np.savetxt(dir_fold + "fpr.csv", fpr)
    np.savetxt(dir_fold + "tpr.csv", tpr)
    np.savetxt(dir_fold + "roc_auc.csv", [roc_auc])
    aucs.append(roc_auc)
    # fpr, tpr, _ = roc_curve(np.array(ts_matches).flatten(), np.array(sim).flatten())
    # roc_auc = auc(fpr, tpr)
    # np.savetxt(dir_results + pair_type + "/" + "fpr.csv", fpr)
    # np.savetxt(dir_results + pair_type + "/" + "tpr.csv", tpr)
    # np.savetxt(dir_results + pair_type + "/" + "roc_auc.csv", [roc_auc])

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
