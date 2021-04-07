from pathlib import Path
from typing import List, Union

import en_core_web_sm
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import SBERT_MODEL_NAME
from utils.types import FolkLoreData, FolkLoreEmb, FolkLoreEmbCoarse

nlp = en_core_web_sm.load()
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)


def get_SBert_avg(doc: str):
    doc = nlp(doc)
    sents: List[str] = []
    for i, token in enumerate(doc.sents):
        sents.append(token.text.strip())
    # (|sents|, 768)
    sents_embeddings = sbert_model.encode(sents)
    return sents_embeddings.mean(axis=0)

def get_LF_avg(doc: str):
    inputs = LF_tokenizer(doc, max_length=4096, truncation=True, return_tensors="pt")
    outputs = LF_model(**inputs)
    return outputs["pooler_output"].detach()[0]


def get_SBert_from_file(path: Path) -> np.ndarray:
    with path.open() as f:
        doc = "".join(f.readlines())
        emb = get_SBert_avg(doc)
    return emb

def get_LF_from_file(path: Path) -> np.ndarray:
    with path.open() as f:
        doc = "".join(f.readlines())
    emb = get_LF_avg(doc)
    return emb


def get_emb_from_dict(folklore_data: FolkLoreData, coarse=False, method="SBert") -> Union[FolkLoreEmb, FolkLoreEmbCoarse]:
    """if coarse, save each individual subtext's embeddings in dictionary
      else: save average"""
    assert method in ["SBert", "LF"]
    folklore2emb = {}
    for folklore, texts in tqdm(folklore_data.items()):
        if not coarse:
            # (| texts |, 768)
            if method == "SBert":
                total_emb = np.stack([get_SBert_from_file(file) for file in tqdm(texts)])
            else:
                total_emb = np.stack([get_LF_from_file(file) for file in tqdm(texts)])
            folklore2emb[folklore] = total_emb.mean(axis = 0)
        else:
            folklore2emb[folklore]: FolkLoreEmb = {}
            for text in tqdm(texts):
                if method == "SBert":
                    folklore2emb[folklore][text] = get_SBert_from_file(text)
                else:
                    folklore2emb[folklore][text] = get_LF_from_file(text)
    return folklore2emb
