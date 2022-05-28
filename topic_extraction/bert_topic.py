import json

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
from utils.data_utils import load_data, print_topic_size

class BertTopicForSummarization():

    def __init__(self,
                 min_topic_size: int = 7,
                 embedding_model : str = "all-MiniLM-L6-v2"):
        self._min_topic_size = min_topic_size
        self._embedding_model = SentenceTransformer(embedding_model)

    def train_model(self,
                    data_split: str,
                    model_file_path: str):
        self.topic_model = BERTopic(verbose=True,
                                    n_gram_range=(3,3),
                                    embedding_model=self._embedding_model,
                                    nr_topics="auto")
        docs = load_data(split=data_split)
        topics, probabilities = self.topic_model.fit_transform(docs)
        self.topic_model.save(model_file_path)
        return topics, probabilities

    def __call__(self,
                 model_file_path="/Users/secilsen/PycharmProjects/LongDocumentSummarization/models/bertopic/model",
                 data_split="train"):
      #  if os.path.exists(model_file_path):
      #      self.topic_model = BERTopic.load(model_file_path)
      #      docs = load_data(split=data_split)
      #      print_topic_size(self.topic_model, docs)
      #  else:
            topics, probabilities = self.train_model(data_split, model_file_path)


