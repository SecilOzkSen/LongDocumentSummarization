from topic_extraction.bert_topic import BertTopicForSummarization
from utils.data_utils import load_data
import re

def new_version():
    train = load_data()
    return train


if __name__ == "__main__":
    docs = load_data(split='train')
    bert_topic = BertTopicForSummarization(docs)
    topic = bert_topic(docs[0])
    print(topic)

  #  docs = load_data(split='train')
  #  bert_topic(docs[0])

