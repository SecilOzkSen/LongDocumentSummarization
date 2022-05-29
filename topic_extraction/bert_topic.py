import json
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
from sklearn.feature_extraction.text import CountVectorizer
from typing import List

IRRELEVANT_TOPICS_LIST = [
    "fig",
    "et al",
    "ref",
    "hep"
]

class BertTopicForSummarization():

    def __init__(self,
                 docs: List[str],
                 min_topic_size: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ):
        self.docs = docs
        self._min_topic_size = min_topic_size
        self._embedding_model = SentenceTransformer(embedding_model)
        self.get_model()

    def train_model(self,
                    model_file_path: str):
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        self.topic_model = BERTopic(verbose=True,
                                    min_topic_size=self._min_topic_size,
                                 #   nr_topics="auto",
                                    vectorizer_model=vectorizer_model
                                    )
        embeddings = self._embedding_model.encode(self.docs, show_progress_bar=True)
        topics, probabilities = self.topic_model.fit_transform(self.docs, embeddings)
        self.topic_model.save(model_file_path)

    def get_model(self,
                  model_file_path="/Users/secilsen/PycharmProjects/LongDocumentSummarization/models/bertopic/model2.bt",
                  data_split="train"):
          if os.path.exists(model_file_path):
              self.topic_model = BERTopic.load(model_file_path, embedding_model=self._embedding_model)
          else:
              self.train_model(model_file_path)

    @staticmethod
    def _topic_validator(topic:str) -> bool:
        if topic in IRRELEVANT_TOPICS_LIST:
            return False
        if len(topic) <= 2:
            return False
        else:
            return True

    def get_topic_as_str(self, predicted_topic_cluster_id):
        topics = self.topic_model.topics
        topics_list = topics.get(predicted_topic_cluster_id)
        for serie in topics_list:
            topic = serie[0]
            if self._topic_validator(topic):
                return topic
        return ""

    def __call__(self, doc: str):
        # get topic for each doc
        topic_predictions, probabilities = self.topic_model.transform([doc])
        return self.get_topic_as_str(topic_predictions[0])






