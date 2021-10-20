import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_data(
        *,
        preprocess=None,
        filter_no=None,
        test_size=0.2,
        random_state=42
):
    """
    Parameters
    ----------
    preprocess list of string, order matters
    filter_no only return ATU with stories >= filter_no
    test_size split test size
    random_state random seed

    Returns
    -------
    X_train, X_test, y_train, y_test

    """
    if preprocess is None:
        preprocess = []
    atu = pd.read_hdf("../datasets/ATU.h5", key="SBERT&LF_MERGE")
    if filter_no:
        assert type(filter_no) is int
        atu = atu.loc[atu.groupby("atu")["atu"].filter(lambda g: len(g) >= filter_no).index]


    X, y = np.array(atu["LF"].tolist()), atu["atu"]

    if preprocess:
        pipe = []
        for p in preprocess:
            if p == "standard_scaler":
                pipe.append((p, StandardScaler()))
            elif p == "pca":
                pipe.append((p, PCA(n_components=30)))
        pipeline = Pipeline(pipe)
        X = pipeline.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=y
                            )





