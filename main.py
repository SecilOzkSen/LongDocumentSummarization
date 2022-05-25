from topic_extraction.bert_topic import BertTopicForSummarization
from utils.data_utils import load_data
import re

def new_version():
    train = load_data()
    return train


if __name__ == "__main__":
    bert_topic = BertTopicForSummarization()
    bert_topic()

