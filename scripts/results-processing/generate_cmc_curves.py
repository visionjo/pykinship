from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.tools.metrics import evaluate


def load_dataframes(din, wildcard="*fusion*.csv"):
    data = {}
    for f in din.glob(wildcard):
        print(f)
        ref = "-".join(f.with_name("").name.split("_")[-3:]).replace("-fusion", "")
        print(ref)
        data[ref] = pd.read_csv(f)


if __name__ == "__main__":
    dir_results = Path(
        Path.home().joinpath(
            "Dropbox/FIW_Video/results/search_retrieval_evaluation/results/image_based"
        )
    )

    df_list = load_dataframes(dir_results)

    result = np.load(
        dir_results.joinpath("gallery_probe_features.npy"), allow_pickle=True
    )

    gallery_feature = torch.FloatTensor(result.item().get("gallery_f"))
    gallery_label = result.item().get("gallery_label")
    print("gallery size:", gallery_feature.size(), gallery_label.shape)

    query_feature = torch.FloatTensor(result.item().get("query_f"))
    query_label = result.item().get("query_label")
    print("query size:", query_feature.size(), query_label.shape)

    query_feature = query_feature.squeeze(1)
    gallery_feature = gallery_feature.squeeze(1)

    ## query-gallery
    CMC = torch.IntTensor(gallery_label.shape[0]).zero_()
    ap = 0.0
    all_scores = []
    all_predicts = []
    for i in range(query_label.shape[0]):
        scores, predicts, (ap_tmp, CMC_tmp) = evaluate(
            query_feature[i], query_label[i], gallery_feature, gallery_label
        )
        all_scores.append(scores.squeeze())
        all_predicts.append(predicts)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / query_label.shape[0]  # average CMC
    print("Rank@1:%f Rank@5:%f Rank@10:%f" % (CMC[0], CMC[4], CMC[9]))
    print("Rank@10:%f Rank@20:%f Rank@50:%f" % (CMC[9], CMC[19], CMC[49]))
    print("mAP:%f" % (ap / query_label.shape[0]))

    # save all_scores to npy
    predict_result = {
        "score": np.asarray(all_scores),
        "predict": np.asarray(all_predicts),
    }
    np.save("predict_result.npy", predict_result)

    CMC = CMC.numpy()
    fig, ax = plt.subplots()
    plt.plot(CMC)
    ax.set(xscale="log")
    plt.xlim(0, 1000)
    plt.show()
    fig.savefig("CMC_result.png")
