import os

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from torch import nn
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    IntervalStrategy, AutoTokenizer
from torch.utils.data import DataLoader
from transformers.data import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_Roberta_avg(doc: str):
    inputs = roberta_tokenizer(doc,
                               max_length=128,
                               truncation=True,
                               padding=True,
                               return_tensors="pt")
    outputs = roberta_model(**inputs)
    return outputs["pooler_output"].detach()[0]




if __name__ == "__main__":
    df: pd.DataFrame = pd.read_hdf("../datasets/ATU.h5", key="default")
    df = df[df["motif"] != "NOT FOUND"]
    atu = df.loc[df.groupby("atu")["atu"].filter(lambda g: len(g) >= 10).index]

    print(atu.columns)
    emb = np.array(atu["Roberta"].tolist())
    label = atu["motif"]

    X_train, X_test, y_train, y_test = train_test_split(emb, label, test_size=0.1, random_state=42)
    clfs = [
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        SVC(kernel="linear", C=0.025),
        KNeighborsClassifier(3)
    ]

    for clf in clfs:
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(type(clf).__name__, accuracy_score(y_test, y_pred))


    # print(emb.shape, label.shape)

    # roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    # roberta_model = RobertaModel.from_pretrained("roberta-large")
    # from tqdm import tqdm
    # tqdm.pandas()
    #
    # df["Roberta"] = df.progress_apply(
    #     lambda row: get_Roberta_avg(row["text"]),
    #     axis=1
    # )
    # df.to_hdf("../datasets/ATU.h5", key="default")



