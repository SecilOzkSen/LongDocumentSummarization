from model.bert_topic import BertTopicForSummarization
from utils.data_utils import load_data

if __name__ == "__main__":
    train, validation, test = load_data()
    bert_topic = BertTopicForSummarization(verbose=True, min_topic_size=5)
    bert_topic(train["article"])