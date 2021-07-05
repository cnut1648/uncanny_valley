from pathlib import Path
from typing import List, Union

import en_core_web_sm
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import gensim
from torch import nn as nn

from config import SBERT_MODEL_NAME
from utils.types import FolkLoreData, FolkLoreEmb, FolkLoreEmbCoarse

nlp = en_core_web_sm.load()
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

from transformers import LongformerModel, LongformerTokenizerFast, LongformerConfig
LFconfig = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
LF_model = LongformerModel.from_pretrained('allenai/longformer-base-4096', config = LFconfig)
LF_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
LF_tokenizer.model_max_length = LF_model.config.max_position_embeddings


class MatrixVectorScaledDotProductAttention(nn.Module):

	def __init__(self, temperature, attn_dropout=0.1):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, q, k, v, mask=None):
		"""
		q: tensor of shape (n*b, d_k)
		k: tensor of shape (n*b, l, d_k)
		v: tensor of shape (n*b, l, d_v)
		returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
		"""
		attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
		attn = attn / self.temperature
		if mask is not None:
			attn = attn.masked_fill(mask, -np.inf)
		attn = self.softmax(attn)
		attn = self.dropout(attn)
		output = (attn.unsqueeze(2) * v).sum(1)
		return output, attn


class MultiheadAttPoolLayer(nn.Module):

	def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
		super().__init__()
		assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
		self.n_head = n_head
		self.d_k = d_k_original // n_head
		self.d_v = d_k_original // n_head

		self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
		self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
		self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

		nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

		self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, mask=None):
		"""
		q: tensor of shape (b, d_q_original)
		k: tensor of shape (b, l, d_k_original)
		mask: tensor of shape (b, l) (optional, default None)
		returns: tensor of shape (b, n*d_v)
		"""
		n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

		bs, _ = q.size()
		bs, len_k, _ = k.size()

		qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
		ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
		vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

		qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
		ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
		vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

		if mask is not None:
			mask = mask.repeat(n_head, 1)
		output, attn = self.attention(qs, ks, vs, mask=mask)

		output = output.view(n_head, bs, d_v)
		output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
		output = self.dropout(output)
		return output, attn

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

def save_Doc2Vec_from_df(df, text_col,
						vector_size=768,
						min_count=5,
						):
	"""
	Parameters
	----------
	df contains text
	text_col col name of text
	vector_size resulting vector size
	min_count ignore too rare words in text

	Returns
	-------

	"""
	corpus = []
	for i, row in tqdm(df.iterrows()):
		tokens = gensim.utils.simple_preprocess(row[text_col])
		corpus.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))
	model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=40)
	model.build_vocab(corpus)
	model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
	model.save("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/gensim")
	model = gensim.models.Doc2Vec.load("/content/drive/MyDrive/Creepy Data/folklores/cleaned data/gensim")

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


