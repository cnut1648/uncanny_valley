from typing import Union

import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.folklores import getTextFolklore
from SBERT import get_SBert_from_dict


def getTSNE(vec: Union[np.ndarray, list], n_components=3):
    if type(vec) is not list:
        assert len(vec.shape) in [1, 2]
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)
    tsne = TSNE(perplexity=30,
                learning_rate=50,
                n_components=n_components,
                n_iter=5000)

    return tsne.fit_transform(vec)


def plot_emb(emb, c_label, c_list=None, title="", projection=True):
    """
  c_label: color label
  c_list: list of numbers, if not provided, generated from 0 - unique(c_label)
  """
    uniq_c_labels: np.ndarray = np.unique(c_label).tolist()
    if c_list is None:
        unique_labels = {
            label: number
            for number, label in enumerate(uniq_c_labels)
        }
        c_list = [unique_labels[label] for label in c_label]
    assert len(c_list) == len(c_label)

    from matplotlib.colors import ListedColormap

    sns.set(style="darkgrid")

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    if projection:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    hsv_modified = plt.cm.get_cmap("hsv", len(c_list))

    colors = ListedColormap(hsv_modified(np.linspace(0, 1, len(c_list))))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if projection:
        ax.set_zlabel("z")
    scatter = ax.scatter(
        *emb.T,
        c=c_list,
        cmap=colors
    )
    plt.legend(handles=scatter.legend_elements()[0], labels=uniq_c_labels)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    from config import DATASET_DIR
    import pandas as pd
    import os

    folklore_data = getTextFolklore(DATASET_DIR)
    folklore2emb = get_SBert_from_dict(folklore_data, coarse=True)

    folklore_df = pd.DataFrame(
        [[folklore, os.path.basename(subtext), emb] for folklore in folklore2emb
         for subtext, emb in folklore2emb[folklore].items()],
        columns=["folklore", "text", "emb"]
    )
    emb_dim_reduced_3d_coarse = getTSNE(folklore_df["emb"].to_list())
    emb_dim_reduced_2d_coarse = getTSNE(folklore_df["emb"].to_list(), 2)

    plot_emb(emb_dim_reduced_3d_coarse,
             folklore_df["folklore"].values,
             title="SBert folklore",
             projection=True)

    plot_emb(emb_dim_reduced_2d_coarse,
             folklore_df["folklore"].values,
             title="SBert folklore",
             projection=False)
