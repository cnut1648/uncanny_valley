"""
for ATU index dataset, scraped from library guides
See libraryguides scraper
"""
from typing import Union, List

import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px

from utils.Embeddings import get_SBert_avg, get_LF_avg, MultiheadAttPoolLayer
from utils.utils import getTSNE


def cleanMFTD(df, MFTD, others: list):
    """
    convert Anthony's MFTD datamodules to library guides

    Parameters
    ----------
    others: hold MFTD's indexes which have more than 2 atu indexes
    """

    atu_mapping = {
        row["atu"]: row["desc"]
        for _, row in df.iterrows()
    }

    newdf = []

    for idx, row in MFTD.iterrows():
        atu = row["ATU"]
        origin = row["Origin"]
        if ',' in atu:
            others.append(idx)
        elif len(atu) >= 6:
            continue
        else:
            newdf.append((
                atu,
                atu_mapping.get(atu, np.nan),
                row["Book Title"],
                origin,
                row["content"],
                row["name"],
                True,
                row["Language"]
            ))
    MFTD = pd.DataFrame(newdf, columns=["atu", "desc", "title", "origin", "text", "url", "from_xml", "language"])

    # postprocess
    # remove redundant atu
    for i, row in MFTD.iterrows():
        atu = row["atu"]
        if atu.startswith("0"):
            atu = atu[1:]
        if atu.endswith("*"):
            atu = atu[:-1]
        if atu != row["atu"]:
            MFTD.at[i, "atu"] = atu
    return MFTD


def read_text_ATU(df, atu: Union[str, List[int]], sample_size: int = 2):
    """
    read `sample_size` from the atu = `atu`

    If used in google colab
    run

    from IPython.display import HTML, display

    def set_css():
      display(HTML('''
      <style>
        pre {
            white-space: pre-wrap;
        }
      </style>
      '''))
    get_ipython().events.register('pre_run_cell', set_css)

    to have text wrapping
    """
    if type(atu) is str:
        x = df.query("atu == @atu").sample(sample_size)["text"]
    else:
        assert len(atu) == 2
        x = df.iloc[atu]["text"]
    print(x.iloc[0])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(x.iloc[1])


def top_k_of_ATU(df, k=10):
    """
    top k most common atu indexes with description
    """
    return df.groupby(["atu", "desc"]).count()["title"].sort_values(ascending=False)[:k]

def emb_ATU_by_attn(df, n_heads = 2, by_emb: str = "LF"):
    """
    for ATU index I, pick random sample story s from I
    pool I into one emb by multi-head self attn
    return df with len |unique(ATU)|
    """
    newdf = []
    for atu in np.unique(df["atu"]):
        I = df[df["atu"] == atu]
        s = I.sample(1).iloc[0]
        desc = s["desc"]
        # (1, 768 )
        s = torch.Tensor(s[by_emb]).unsqueeze(0)
        _, d_q_original = s.shape
        # (1, #stories, 768)
        I = torch.Tensor(I[by_emb].tolist()).unsqueeze(0)
        _, len_q, d_k_original = I.shape

        pooler = MultiheadAttPoolLayer(
            n_heads, d_q_original, d_k_original
        )

        # (1, n_head * d_v = d_k)
        # (n_head * 1, #stories)
        ATU_emb, attn_weights = pooler(s, I, mask=None)
        newdf.append([atu, desc, ATU_emb.detach().squeeze().numpy(), attn_weights.detach().numpy()])
    return pd.DataFrame(newdf, columns=["atu", "desc", "emb", "attn_weights"])

def get_ATU_motif(index: Union[str, int]):
    if type(index) is str:
        try:
            index = int(index)
        except:
            return "NOT FOUND"
    if 1 <= index <= 299:
        return "Animal Tales"
    elif 300 <= index <= 749:
        return "Tales of Magic"
    elif 750 <= index <= 849:
        return "Religious Tales"
    elif 850 <= index <= 999:
        return "Realisitc Tales"
    elif 1000 <= index <= 1169:
        return "TALES OF THE STUPID OGRE (GIANT, DEVIL)"
    elif 1200 <= index <= 1999:
        return "Anecdotes and Jokes"
    elif 2000 <= index <= 2399:
        return "Formula Tales"
    return "NOT FOUND"

def extract_stories_with_valid_ATU(df):
    def has_valid_ATU_motif(str):
        try:
            return True if get_ATU_motif(int(str)) != "NOT FOUND" else False
        except:
            return False

    return df[df["atu"].apply(has_valid_ATU_motif)]

def tsne_visualization(df=None,
                       useHDF5=True,
                       filter_no: int = 10,
                       emb_ls=None,
                       saved_h5_path="path",
                       save_key="SBERT&LF",
                       load_key=None,
                       use_motif=False,
                       perplexities=None):
    #    filter_no
    """
    run TSNE with different perplexities and visualize by plotly
    Parameters
    ----------
    df df with emb col, save different perplexities 2d array to (saved_h5_poth, key=save_key)
    useHDF5 if true, no need df, but load from (saved_h5_poth, key=load_key)
    filter_no only shows ATU indexes with number of stories > filter_no
    emb_ls list of cols that are emb eg. ["LF, "SBERT"]
    use_motif if true, the color is motif, if not the color is ATU individual index
    perplexities list of perplexities to try

    Returns
    -------
    None, visualize
    """
    # can't be all None
    if emb_ls is None:
        emb_ls = ["emb"]
    if load_key is None:
        assert save_key is not None
        load_key = save_key

    if perplexities is None:
        perplexities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    if useHDF5:
        df_tsne = pd.read_hdf(saved_h5_path, key=load_key)
    else:
        assert df is not None
        df_tsne = df.copy()
        for perplexity in tqdm(perplexities):
            for emb in emb_ls:
                emb_2d = getTSNE(df[emb].to_list(), 2, perplexity=perplexity)
                emb_2d = np.array(emb_2d[:, np.newaxis]).tolist()
                df_tsne = pd.concat([df_tsne, pd.DataFrame(emb_2d, columns=[f"{emb}_{perplexity}"])], axis=1)
        df_tsne.to_hdf(saved_h5_path, key=save_key)

    # filter atu with > `filter_no` stories
    if filter_no > 0:
        df_tsne = df_tsne.loc[df_tsne.groupby("atu")["atu"].filter(lambda g: len(g) > filter_no).index]
    if use_motif:
        color = df_tsne["motif"]
    else:
        color = df_tsne["atu"] + "(" + df_tsne["desc"] + ")"

    for perplexity in perplexities:
        for emb in emb_ls:
            embeddings = np.array(df_tsne[f"{emb}_{perplexity}"].tolist())
            fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], color=color, title=f"{emb}_{perplexity}")
            fig.show()


def load_ATU(useHDF5=True):
    """ TODO
    merge MFTD & df
     """
    if useHDF5:
        df = pd.read_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned datamodules/ATU.h5", key="SBERT&LF_MERGE")

    else:
        # processing from source and save as h5

        # library guides
        df = pd.read_json("/content/drive/MyDrive/Creepy Data/folklores/cleaned datamodules/ATU.jl", lines=True)
        df[["LF", "SBERT"]] = df.apply(lambda row: (
            get_SBert_avg(row["text"]),
            get_LF_avg(row["text"])
        ), axis=1, result_type="expand")
        df.at[:, "language"] = "language"
        df.at[:, "from_xml"] = False
        df.to_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned datamodules/ATU.h5", key="SBERT&LF")

        # MFTD
        MFTD = pd.read_csv("/content/drive/MyDrive/Creepy Data/folklores/cleaned datamodules/MFTD.csv")
        # TODO
        # right now ignore others
        others = []
        MFTD = cleanMFTD(df, MFTD, others)
        # retain only English text
        MFTD = MFTD[MFTD["language"] == "English"]
        MFTD[["LF", "SBERT"]] = MFTD.apply(lambda row: (
            get_SBert_avg(row["text"]),
            get_LF_avg(row["text"])), axis=1, result_type="expand")
        MFTD.to_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned datamodules/MFTD.h5", key="SBERT&LF")

        df = pd.concat([df, MFTD])
        df.reset_index(drop=True, inplace=True)
        df.to_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned datamodules/ATU.h5",
                  key="SBERT&LF_MERGE")

        print(df.sample(3))
        print(MFTD.sample(3))
        print("Same ATU indexes after processing")
        print(len(
            set(MFTD.query("language == 'English'")["atu"]).intersection(set(df["atu"]))
        ))


if __name__ == '__main__':
    useHDF5 = True
    df = load_ATU(useHDF5=useHDF5)
