"""
Perform template adaptation for the test split of Track I. 

Usage: template_adaptation_v2.py <verification-lists> <visual-features> [options]

Options:
    --balanced
    --normalize
    --audio=<audio-path>

Docs
--------------
<verification-lists> should be the path to a folder containing {train.csv, test.csv, val.csv}.
<visual-features> should be the path to a folder containing the visual features to be used.

If --balanced is passed in, the classes will be reweighted during SVM training.
If --normalize is passed in, the features will be unit normalize.

To use audio features, use the --audio flag and provide a path to the audio features.
"""
from pathlib import Path
from tqdm.auto import tqdm
from docopt import docopt
import pandas as pd
import numpy as np
import random
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

random.seed(42)


def make_negatives(val, train, features, audio_features=None):
    """
    Make the set of negative examples that each SVM
    will use.

    Returns
    --------
    A (num_mids x 512) shaped array of features.
    """
    combined = pd.concat([val, train])
    mids = np.hstack([combined.p1, combined.p2])
    mids = set(mids)

    print("Picking one encoding for each negative MID.")
    negative_features = []
    for mid in tqdm(mids):
        with open(features / mid / "encodings.pkl", "rb") as f:
            mid_features = pickle.load(f)

        if audio_features:
            npy_files = (audio_features / mid).glob("*.npy")
            d_vectors = [np.load(_) for _ in npy_files]
        else:
            d_vectors = []

        chosen_feature = random.choice(list(mid_features.values()) + d_vectors)
        negative_features.append(chosen_feature)

    return np.vstack(negative_features)

def adapt_template(mid, negative_features, features, normalize=False, balanced=False, audio_features=None):
    """
    Train a one-v-rest SVM for an mid that knows how to discriminate
    between the individual and other.

    Returns
    ----------
    clf: The SVM trained for the MID.
    mid_features: The positive features the MID was trained on.
    """

    # Load the features for this person.
    with open(features / mid / "encodings.pkl", "rb") as f:
        mid_features = pickle.load(f)

    if audio_features:
        npy_files = (audio_features / mid).glob("*.npy")
        d_vectors = [np.load(_) for _ in npy_files]
    else:
        d_vectors = []

    mid_features = np.vstack(list(mid_features.values()) + d_vectors)
    mid_features = mid_features

    Y = np.concatenate(
        [np.ones(len(mid_features)), np.zeros(len(negative_features))]
    )
    X = np.vstack([mid_features, negative_features])

    if normalize:
        X = preprocessing.normalize(X)

    clf = svm.SVC(C=10, kernel="rbf", class_weight="balanced" if balanced else None)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    acc = accuracy_score(Y, Y_pred)
    print(f"Accuracy: {acc}")
    return clf, mid_features



if __name__ == "__main__":

    args = docopt(__doc__)
    lists = Path(args["<verification-lists>"])
    visual_features = Path(args["<visual-features>"])
    balanced = args["--balanced"]
    normalize = args["--normalize"]
    audio_features = args["--audio"]
    audio_features = Path(audio_features) if audio_features else None


    output_name = f"templates/{visual_features.name}"
    if normalize:
        output_name += "_normalized"
    if balanced:
        output_name += "_balanced"
    if audio_features:
        output_name += "_withaudio"

    templates_save_path = Path(output_name)
    templates_save_path.mkdir(exist_ok=True)


    # Load all the splits. Although we'll only be doing template
    # adaptation on the test split, negative samples will be drawn
    # from the train and validation splits. 
    test = pd.read_csv(lists / "test.csv")
    val = pd.read_csv(lists / "val.csv")
    train = pd.read_csv(lists / "train.csv")
    

    negative_features = make_negatives(val, train, visual_features, audio_features)
    mids_in_test = set(np.hstack([test.p1, test.p2]))
    svm_bank = {}
    mid_feature_bank = {}
    for mid in tqdm(mids_in_test):
        svm_for_mid, mid_features = adapt_template(mid, negative_features, visual_features, normalize=normalize, balanced=balanced, audio_features=audio_features)
        svm_bank[mid] = svm_for_mid
        mid_feature_bank[mid] = mid_features
    
    with open(templates_save_path / "svm_bank.pkl", "wb") as f:
        pickle.dump(svm_bank, f)

    with open(templates_save_path / "mid_feature_bank.pkl", "wb") as f:
        pickle.dump(mid_feature_bank, f)