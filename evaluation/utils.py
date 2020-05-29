import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from sklearn.metrics.pairwise import cosine_similarity

#######################################################################
# Evaluate
def compute_scores(feature_seta, feature_setb):
    query = feature_seta.view(-1, 1)

    x1 = feature_setb.cpu().numpy()
    x2 = query.cpu().numpy().reshape(1, -1)
    return cosine_similarity(x1, x2, dense_output=True)


def make_prediction(scores):
    # predict index
    index = np.argsort(scores.squeeze(1))  # from small to large
    return index[::-1]


def evaluate(features_probe, labels_probes, features_gallery, labels_gallery):
    """
    Does end-to-end evaluation. Computes CMC
    :param features_probe:
    :param labels_probes:
    :param features_gallery:
    :param labels_gallery:
    :return:    CMC value?
    """
    pass

    scores = compute_scores(features_probe, features_gallery)

    ranked_list_predicted = make_prediction(scores)

    list_true_relatives = np.argwhere(labels_gallery == labels_probes)

    cmc_tmp = compute_mAP(ranked_list_predicted, list_true_relatives)

    return scores, ranked_list_predicted, cmc_tmp


def compute_mAP(predicted_indices, true_indices):
    ap = 0
    cmc = torch.IntTensor(len(predicted_indices)).zero_()
    if not true_indices.size:  # if empty
        cmc[0] = -1
        return ap, cmc

    # find good_index index
    ngood = len(true_indices)
    mask = np.in1d(predicted_indices, true_indices)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


if __name__ == '__main__':
    ######################################################################    
    result = np.load("gallery_probe_features.npy")

    gallery_feature = torch.FloatTensor(result.item().get('gallery_f'))
    gallery_label = result.item().get('gallery_label')
    print("gallery size:", gallery_feature.size(), gallery_label.shape)

    query_feature = torch.FloatTensor(result.item().get('query_f'))
    query_label = result.item().get('query_label')
    print("query size:", query_feature.size(), query_label.shape)

    query_feature = query_feature.cuda().squeeze(1)
    gallery_feature = gallery_feature.cuda().squeeze(1)

    ## query-gallery
    CMC = torch.IntTensor(gallery_label.shape[0]).zero_()
    ap = 0.0
    all_scores = []
    all_predicts = []
    for i in range(query_label.shape[0]):
        scores, predicts, (ap_tmp, CMC_tmp) = evaluate(query_feature[i], query_label[i],
                                   gallery_feature, gallery_label)
        all_scores.append(scores.squeeze())
        all_predicts.append(predicts)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / query_label.shape[0]  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f' % (CMC[0], CMC[4], CMC[9]))
    print('Rank@10:%f Rank@20:%f Rank@50:%f' % (CMC[9], CMC[19], CMC[49]))
    print('mAP:%f' % (ap / query_label.shape[0]))

    # save all_scores to npy
    predict_result = {'score': np.asarray(all_scores), 'predict': np.asarray(all_predicts)}
    np.save("predict_result.npy", predict_result)

    CMC = CMC.numpy()
    fig, ax = plt.subplots()
    plt.plot(CMC)
    ax.set(xscale="log")
    plt.xlim(0,1000) 
    plt.show()
    fig.savefig('CMC_result.png')

