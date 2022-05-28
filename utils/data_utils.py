import torch
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
    txt = re.sub(r'[^a-zA-Z0-9 .,\'\n]', "", txt).replace("  ", " ").strip()
    txt = re.sub(r'xcite*?', '', txt)
    txt = re.sub(r'xmath*?', '', txt)
    txt = re.sub(r'h[0-9]+', '', txt)
    txt_tokens = word_tokenize(txt)
    filtered_sentence = [w for w in txt_tokens if not w.lower() in stop_words]
    sentence = ' '.join(filtered_sentence)
    return sentence

def load_data_from_file(data_file_path, is_gold_summaries):
    with open(data_file_path, 'r') as fp:
        json_str = json.load(fp)
    if is_gold_summaries:
        return get_data_as_list(json_str, "article"), get_data_as_list(json_str, "abstract")

    return get_data_as_list(json_str, "article")

def get_data_as_list(json_str, field = "article"):
    data_list = []
    obj = json.loads(json_str)
    for _fields in obj:
        data_list.append(_fields[field])
    return data_list

def load_data_as_torch(split='train',
                       is_gold_summaries=False,
                       data_file_path = "/Users/secilsen/PycharmProjects/LongDocumentSummarization/data"):

    data_file_path = f"{data_file_path}/{split}.json"
    print("Data loading....")

    if os.path.exists(data_file_path):
        print("Data loading ends.")
        return load_data_from_file(data_file_path, is_gold_summaries)

    dataset = load_dataset("ccdv/arxiv-summarization", split=split)
    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.sample(frac=0.3)
    df_dataset["article"] = df_dataset["article"].apply(lambda x: cleanup(x))
    df_dataset["abstract"] = df_dataset["abstract"].apply(lambda x: cleanup(x))

    js = df_dataset.to_json(orient='records')

    with open(data_file_path, 'w') as fp:
        json.dump(js, fp)

    _x = df_dataset["article"].values.tolist()
    print("Data loading ends.")

    if is_gold_summaries:
        _y = dataset["abstract"]
        return torch.tensor(_x, dtype=torch.stri), _y

    return _x

def load_data(split='train',
              is_gold_summaries=False,
              data_file_path = "/Users/secilsen/PycharmProjects/LongDocumentSummarization/data"):
    data_file_path = f"{data_file_path}/{split}.json"
    print("Data loading....")

    if os.path.exists(data_file_path):
        print("Data loading ends.")
        return load_data_from_file(data_file_path, is_gold_summaries)

    dataset = load_dataset("ccdv/arxiv-summarization", split=split)
    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.sample(frac=0.3)
    df_dataset["article"] = df_dataset["article"].apply(lambda x: cleanup(x))
    df_dataset["abstract"] = df_dataset["abstract"].apply(lambda x: cleanup(x))

    js = df_dataset.to_json(orient='records')

    with open(data_file_path, 'w') as fp:
        json.dump(js, fp)

    _x = df_dataset["article"].values.tolist()
    print("Data loading ends.")

    if is_gold_summaries:
        _y = dataset["abstract"]
        return _x, _y

    return _x

## Util methods for BERTopic visualisation
def print_topic_size(topic_model: BERTopic, docs):
    print(topic_model.topics)
    topic_model.visualize_barchart()
    new_topics = topic_model.reduce_topics(docs, topic_model.topics)
    print(new_topics)












