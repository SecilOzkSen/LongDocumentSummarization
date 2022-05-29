from topic_extraction.bert_topic import BertTopicForSummarization
from utils.data_utils import load_data, load_data_v2
import re
import numba

def new_version():
    train = load_data()
    return train


if __name__ == "__main__":
    df_dataset = load_data_v2('train')
    numba.cuda.profile_stop()
 #   docs = load_data(split='train')
 #   bert_topic = BertTopicForSummarization(docs)
 #   topic = bert_topic(docs[0])
 #   print(topic)



  #  docs = load_data(split='train')
  #  bert_topic(docs[0])

