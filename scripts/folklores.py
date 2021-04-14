"""
for folklores (Most consists of mythology
"""
import os
from pathlib import Path
import pandas as pd

import h5py as h5py

from config import DATASET_DIR
from utils.Embeddings import get_emb_from_dict
from utils.types import FolkLoreData
from utils.utils import getTSNE, plot_embed, top_k_most_similar


def getTextFolklore(path: Path) -> FolkLoreData:
    """
    get text path from directories
    not loading so that save memory
    """
    folklore_data = {}
    for book in os.listdir(path):
        text_for_book = []
        for text in os.listdir(path / book):
            if text.lower().startswith("split "):
                text_for_book = [text]
                continue
            if text != ".ipynb_checkpoints":
                text_for_book.append(path / book / text)

        # postprocess book that has split by
        first = text_for_book[0]
        if str(first).lower().startswith("split "):
            text_for_book = [path / book / first / file for file in os.listdir(path / book / first) if
                             file != ".ipynb_checkpoints"]
        folklore_data[book] = text_for_book

    return folklore_data

def case_study(folklore_df, emb = "LF", method="plot"):
    """
    look at a few stories related to flood
    and 1. if plot: see there location in t-SNE
        2. if simialr: compute most similar stories in terms of cos metric
    """

    assert method in ["plot", "similar"]

    if method == "plot":
        emb_reduced_2d = []
        perplexities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
        for perplexity in perplexities:
            emb_reduced_2d.append(getTSNE(folklore_df[emb].to_list(), 2, perplexity))

        for emb, perplexity in zip(emb_reduced_2d, perplexities):
            gilgamesh = folklore_df.query(
                "folklore=='The Epic of Gilgamesh' and text=='TABLET XI THE STORY OF THE FLOOD.txt'").index.values
            gilgamesh_2d = emb[gilgamesh].reshape(-1)
            eridu_genesis = folklore_df.query("folklore=='Eridu Genesis' and text=='The Flood.txt'").index.values
            eridu_genesis_2d = emb[eridu_genesis].reshape(-1)
            genesis = folklore_df.query(
                "folklore=='The Old Testament of the King James Version of the Bible' and text=='Noah and the flood'").index.values
            genesis_2d = emb[genesis].reshape(-1)
            plot_embed(emb,
                       folklore_df["folklore"].values,
                       title=f"SBert folklore perplexity={perplexity}",
                       projection=False)
            texts = [
                plt.text(*arr, s, ha="center", va="center")
                for (arr, s) in zip([gilgamesh_2d, eridu_genesis_2d, genesis_2d], ["Gilgamesh", "Eridu Genesis", "Genesis"])
            ]
            adjust_text(texts)
    else:
        gilgamesh = folklore_df.query(
            "folklore=='The Epic of Gilgamesh' and text=='TABLET XI THE STORY OF THE FLOOD.txt'").index.values
        eridu_genesis = folklore_df.query("folklore=='Eridu Genesis' and text=='The Flood.txt'").index.values
        genesis = folklore_df.query(
            "folklore=='The Old Testament of the King James Version of the Bible' and text=='Noah and the flood'").index.values

        cos_matrix = util.pytorch_cos_sim(folklore_df[emb], folklore_df[emb])
        print(
            folklore_df.iloc[top_k_most_similar(cos_matrix, gilgamesh, 10)]
        )
        print(
            folklore_df.iloc[top_k_most_similar(cos_matrix, eridu_genesis, 10)]
        )
        print(
            folklore_df.iloc[top_k_most_similar(cos_matrix, genesis, 10)]
        )


def load_folklore(useHDF5=True):
    with h5py.File("/content/drive/MyDrive/Creepy Data/folklores/folklore_noATU.h5", "a") as f:
        print(f.keys())

    if useHDF5:
        folklore_df = pd.read_hdf("/content/drive/MyDrive/Creepy Data/folklores/folklore_noATU.h5", key="LF&SBERT")
    else:
        folklore_data = getTextFolklore(DATASET_DIR)

        folklore_data.update(getTextFolklore(DATASET_DIR / "The King James Version of the Bible"))
        del folklore_data["The King James Version of the Bible"]
        # TODO
        del folklore_data["Folklore and Mythology Electronic Texts"]

        for book in folklore_data:
            print(book)
            print("\t", [str(os.path.basename(p)) for p in folklore_data[book]])

        folklore2emb_SBERT = get_emb_from_dict(folklore_data, coarse=True, method="SBERT")
        folklore2emb_LF = get_emb_from_dict(folklore_data, coarse=True, method="LF")

        folklore_df_LF = pd.DataFrame(
            [[folklore, os.path.basename(subtext), subtext, emb] for folklore in folklore2emb_LF
             for subtext, emb in folklore2emb_LF[folklore].items()],
            columns=["folklore", "text", "path", "LF"]
        )
        folklore_df_SBert = pd.DataFrame(
            [[folklore, os.path.basename(subtext), subtext, emb] for folklore in folklore2emb_SBERT
             for subtext, emb in folklore2emb_SBERT[folklore].items()],
            columns=["folklore", "text", "path", "SBert"]
        )

        folklore_df = pd.merge(folklore_df_SBert, folklore_df_LF)
        folklore_df.to_hdf("/content/drive/MyDrive/Creepy Data/folklores/folklore_noATU.h5", key="LF&SBERT")

    return folklore_df


if __name__ == "__main__":
    folklore_df = load_folklore(True)
    print(folklore_df.sample(5))
