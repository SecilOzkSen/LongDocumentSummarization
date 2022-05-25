from datasets import load_dataset
import json
from constants import (TRAIN_FILE_PATH,
                       VALIDATION_FILE_PATH,
                       TEST_FILE_PATH,
                       FULL_FILE_PATH)
import re
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os

stop_words = set(stopwords.words('english'))

def load_dataset_from_hub():
    dataset = load_dataset("ccdv/arxiv-summarization")
    whole_dataset = dataset.to_dict()
    train_dict = dataset["train"].to_dict()
    validation_dict = dataset["validation"].to_dict()
    test_dict = dataset["test"].to_dict()
    with open(FULL_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(whole_dataset, f)
    with open(TRAIN_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_dict, f)
    with open(TRAIN_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_dict, f)
    with open(VALIDATION_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(validation_dict, f)
    with open(TEST_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_dict, f)
    return get_dataset_splits()

def get_dataset_splits():
    with open(TRAIN_FILE_PATH, "r", encoding='utf-8') as f:
        train_data = json.load(f)
    with open(VALIDATION_FILE_PATH, "r", encoding='utf-8') as f:
        validation_data = json.load(f)
    with open(TEST_FILE_PATH, "r", encoding='utf-8') as f:
        test_data = json.load(f)
    return train_data, validation_data, test_data

def cleanup(txt:str):
    txt = re.sub(r'[^a-zA-Z0-9 .,\n]', "", txt).replace("  ", " ").strip()
    txt = re.sub('xcite*?', '', txt)
    txt = re.sub('xmath*?', '', txt)
    txt_tokens = word_tokenize(txt)
    filtered_sentence = [w for w in txt_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def load_data_from_file(data_file_path):
    with open(data_file_path, 'r') as fp:
        json_str = json.load(fp.read())
    return json_str

def load_data(split='train',
              is_gold_summaries=False,
              data_file_path = "/Users/secilsen/PycharmProjects/LongDocumentSummarization/data"):
    data_file_path = f"{data_file_path}/{split}.json"
    print("Data loading....")

    if os.path.exists(data_file_path):
        return load_data_from_file(data_file_path)

    dataset = load_dataset("ccdv/arxiv-summarization", split=split)
    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset["article"].apply(lambda x: cleanup(x))
    df_dataset = df_dataset["abstract"].apply(lambda x: cleanup(x))

    js = df_dataset.to_json(orient='records')

    with open(data_file_path, 'w') as fp:
        json.dump(js, fp)

    _x = df_dataset["article"].values.to_list()
    print("Data loading ends.")

    if is_gold_summaries:
        _y = dataset["abstract"]
        return _x, _y

    return _x

## Util methods for BERTopic visualisation
def print_topic_size(topic_model: BERTopic):
    print(topic_model.topics)












