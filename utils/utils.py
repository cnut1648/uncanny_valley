from typing import Union

import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.folklores import getTextFolklore
from matplotlib.colors import ListedColormap


def getTSNE(vec: Union[np.ndarray, list], n_components=3, perplexity=30):
    if type(vec) is not list:
        assert len(vec.shape) in [1, 2]
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)
    tsne = TSNE(perplexity=perplexity,
                learning_rate=50,
                n_components=n_components,
                n_iter=5000)

    return tsne.fit_transform(vec)


def plot_embed(emb, c_label, c_list=None, title="", projection=True):
    """
    c_label: color label
    c_list: list of numbers, if not provided, generated from 0 - unique(c_label)
    """
    uniq_c_labels = np.unique(c_label).tolist()
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
    hsv_modified = plt.cm.get_cmap("Set3", len(uniq_c_labels))

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
    return ax


def write_format_embeddings_comparators(df, indexes=None):
    """
    https://arxiv.org/pdf/1912.04853.pdf
    vector file is vector
    metafile is index
    map ith row of meta(index) to ith row of vector

    df: dataframe with columan name LF & SBERT
    groupby indexes
    """
    if indexes is None:
        indexes = ["LF", "SBERT"]
    with open("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/vectors_SBERT.tsv", "w") as f_vec_SBERT:
        with open("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/vectors_LF.tsv", "w") as f_vec_LF:
            with open("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/atu.tsv", "w") as f_atu:
                gb = df.groupby(indexes)
                for group in gb.groups:
                    r = gb.get_group(group)
                    for emb, f in zip(["LF", "SBERT"], [f_vec_LF, f_vec_SBERT]):
                        f.write("\t".join(map(str, r[emb].mean())))
                        f.write("\n")
                    f_atu.write(group[0] + "(" + group[1] + ")")
                    f_atu.write("\n")


def top_k_most_similar(cos_matrix, idx, k=10):
    """
    return len-k indices of k most similar
    i.e. for cos metric, the best is 1
    choose the top-k, no order
    """
    return np.argpartition(cos_matrix[idx].flatten().numpy(), -k)[-k:]

def load_motif_mapping(path="datasets/label-mapping.txt") -> dict:
    ret = {}
    with open(path) as f:
        for line in f:
            motif, label = line.split("->")
            ret[int(label)] = motif
    return ret

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
             title="SBert folklore 3D",
             projection=True)

    plot_emb(emb_dim_reduced_2d_coarse,
             folklore_df["folklore"].values,
             title="SBert folklore 2D",
             projection=False)
