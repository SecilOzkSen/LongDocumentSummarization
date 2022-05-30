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
    """
    The BERTopic class to add topic awareness to the summarization process.

    """

    def __init__(self,
                 docs: List[str],
                 min_topic_size: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 model_path: str = "/Users/secilsen/PycharmProjects/LongDocumentSummarization/models/bertopic/model2.bt"
                 ):
        """

        :param docs: List[str], the documents to be classified
        :param min_topic_size: int, default 50, minimum number of topics in the topic model
        :param embedding_model: str, the embedding model utilized by BERTopic
        """

        self._min_topic_size = min_topic_size
        self._embedding_model = SentenceTransformer(embedding_model)
        self._model_file_path = model_path
        self.get_model()

    def train_model(self):
        """
        Trains BERTopic for the input documents and saves it. Returns the topics and respective probabilities.

        """
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        self.topic_model = BERTopic(verbose=True,
                                    min_topic_size=self._min_topic_size,
                                 #   nr_topics="auto",
                                    vectorizer_model=vectorizer_model
                                    )
        embeddings = self._embedding_model.encode(self.docs, show_progress_bar=True)
        topics, probabilities = self.topic_model.fit_transform(self.docs, embeddings)
        self.topic_model.save(self._model_file_path)

    def get_model(self,
                  data_split="train"):
          """
          Load a pretrained topic model from file, if the path is invalid train a new one.

          :param model_file_path: path to the pretrained model
          """

          if os.path.exists(self._model_file_path):
              self.topic_model = BERTopic.load(self._model_file_path, embedding_model=self._embedding_model)
          else:
              self.train_model(self._model_file_path)

    @staticmethod
    def _topic_validator(topic:str) -> bool:
        """
        Returns false if a topic is too short or irrelevant.

        :param topic: str, the topic to be checked
        :return: boolean
        """
        if topic in IRRELEVANT_TOPICS_LIST:
            return False
        if len(topic) <= 2:
            return False
        else:
            return True

    def get_topic_as_str(self, predicted_topic_cluster_id):
        """
        Retrieve the most likely topic, return an empty string if the topic is too short or irrelevant.

        :param predicted_topic_cluster_id: the prediction of topics
        :return: the most probable topic or an empty string if the topic is invalid
        """
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






