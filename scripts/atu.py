"""
for ATU index dataset, scraped from library guides
See libraryguides scraper
"""
import pandas as pd
import numpy as np

from utils.Embeddings import get_SBert_avg, get_LF_avg


def cleanMFTD(df, MFTD, others: list):
    """
    convert Anthony's MFTD data to library guides

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


def read_text_ATU(df, atu: str, sample_size: int = 2):
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
    x = df.query("atu == @atu").sample(sample_size)["text"]
    print(x.iloc[0])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(x.iloc[1])


def top_k_of_ATU(df, k=10):
    """
    top k most common atu indexes with description
    """
    return df.groupby(["atu", "desc"]).count()["title"].sort_values(ascending=False)[:k]

def load_ATU(useHDF5 = True):
    """ TODO
    merge MFTD & df
     """
    pass


if __name__ == '__main__':
    useHDF5 = True

    if useHDF5:
        df = pd.read_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/ATU.h5", key="SBERT&LF")
        MFTD = pd.read_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/MFTD.h5", key="SBERT&LF")

    else:
        # processing from source and save as h5

        # library guides
        df = pd.read_json("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/ATU.jl", lines=True)
        df[["LF", "SBERT"]] = df.apply(lambda row: (
            get_SBert_avg(row["text"]),
            get_LF_avg(row["text"])), axis=1, result_type="expand")
        df.to_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/ATU.h5", key="SBERT&LF")

        # MFTD
        MFTD = pd.read_csv("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/MFTD.csv")
        # TODO
        # right now ignore others
        others = []
        MFTD = cleanMFTD(df, MFTD, others)
        MFTD.to_hdf("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/MFTD.h5", key="SBERT&LF")

    print(df.sample(3))
    print(MFTD.sample(3))
    print("Same ATU indexes after processing")
    print(len(
        set(MFTD.query("language == 'English'")["atu"]).intersection(set(df["atu"]))
    ))
