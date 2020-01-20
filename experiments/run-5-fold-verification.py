import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# from html4vision import Col, imagetable
from src.utils import mkdir

do_scores = False
do_html=False

dir_data = '../data/raw/v0.1.2/'
dir_interim = '../data/dir_interim/'
f_datalist = f'{dir_data}lists/verification_pairs_list_5_fold.pkl'
f_features = f'{dir_interim}features-sphereface-off-the-shelf.pkl'

dir_out = '../results/verification/off-the-shelf-sphereface/'

datatable = pd.read_pickle(f_datalist)

mkdir(dir_out)
if do_scores or 'score' not in datatable:
    features = pd.read_pickle(f_features)

    features = {k.replace('../', ''): v for k, v in features.items()}
    datatable['score'] = datatable.apply(lambda row: np.dot(features[row['p1']], features[row['p2']]), axis=1)

ts_matches = []
sim = []
thresholds = np.arange(datatable.score.values.min(), datatable.score.values.max(), 100)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

folds = datatable.fold.unique()
for fold in folds:
    ids_train = datatable.fold.astype(np.int) != fold
    ids_test = datatable.fold.astype(np.int) == fold

    print(f"Fold: {fold}/{len(folds)}")

    dir_fold = f"{dir_out}fold{fold}/"
    mkdir(dir_fold)

    scores = datatable.loc[ids_test, 'score'].values
    labels = datatable.loc[ids_test, 'label'].values.astype(np.int)

    p1 = datatable.loc[ids_test, 'p1'].values
    p2 = datatable.loc[ids_test, 'p2'].values
    t1 = datatable.loc[ids_test, 'type'].values

    ts_matches.append(labels)
    sim.append(scores)

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(labels, scores)
    ids = np.argsort(scores)
    scores = scores[ids]
    labels = labels[ids]
    p1 = p1[ids]
    p1 = p1[ids]

    t1 = t1[ids]

    roc_auc = auc(fpr, tpr)
    print(f"Acc:{roc_auc}")
    print('Saving Results')
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    mkdir(dir_fold)
    np.savetxt(f"{dir_fold}fpr.csv", fpr)
    np.savetxt(f"{dir_fold}tpr.csv", tpr)
    with open(f"{dir_fold}roc_auc.csv", 'w') as f:
        f.write(f"{roc_auc}")
    aucs.append(roc_auc)


np.savetxt(dir_out + 'auc_scores.csv', aucs, delimiter=',')
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
