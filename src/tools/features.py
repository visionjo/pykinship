from sklearn import preprocessing


def l2_norm(input, axis=1):
    return preprocessing.normalize(input, norm="l2", axis=axis)
