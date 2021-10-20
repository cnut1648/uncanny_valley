from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk
import pandas as pd
from typing import List


nltk.download('stopwords')

atu = pd.read_hdf("/home/jiashu/uncanny_valley/datasets/ATU.h5", key="SBERT&LF_MERGE")

doc: List[str] = [
    line.strip() for line in atu["text"]
]
sp = WhiteSpacePreprocessing(doc, stopwords_language="english")
# use unpreprocess for contextual
preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

print(preprocessed_documents[:2])

tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")

training_dataset = tp.fit(
    text_for_contextual=unpreprocessed_corpus, 
    text_for_bow=preprocessed_documents)

print(tp.vocab[:10])

# 7 topics
ctm = CombinedTM(bow_size=len(tp.vocab), 
    contextual_size=768, n_components=7, 
    num_epochs=20)
ctm.fit(training_dataset) # run the module

print(ctm.get_topic_lists(7))

ctm.save(models_dir="/home/jiashu/uncanny_valley/datasets/ctm")