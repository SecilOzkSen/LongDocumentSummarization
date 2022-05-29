from datasets import load_dataset
import json
import re
import pandas as pd
import os
from topic_extraction.bert_topic import BertTopicForSummarization


def cleanup(txt:str):
    """
    Cleans a text span by removing latex expressions and placeholders.
    :param txt: str, the span to be cleaned
    :return: the cleaned span
    """
    txt = re.sub(r'[^a-zA-Z0-9 .,\'\n]', "", txt).replace("  ", " ").strip()
    txt = re.sub(r'.*?xcite.*?', '', txt)
    txt = re.sub(r'.*?xmath.*?', '', txt)
    txt = re.sub(r'h[0-9]+', '', txt)
    return txt

def load_data_from_file(data_file_path, is_gold_summaries):
    """
    Loads the data from a json file and returns it as a list of strings

    :param data_file_path: the path to the data. The file should be in json format with the data fields article and abstract.
    :param is_gold_summaries: boolean, whether the dataset contains abstracts which are used as gold summaries
    :return: the documents as a list of strings
    """
    with open(data_file_path, 'r') as fp:
        json_str = json.load(fp)
    if is_gold_summaries:
        return get_data_as_list(json_str, "article"), get_data_as_list(json_str, "abstract")

    return get_data_as_list(json_str, "article")

def get_data_as_list(json_str, field = "article"):
    """
    Convert json data into a list of strings

    :param json_str: the json data to be converted
    :param field: which field to convert
    :return: a list of strings
    """
    data_list = []
    obj = json.loads(json_str)
    for _fields in obj:
        data_list.append(_fields[field])
    print("Data loading ends.")

    return data_list

def load_data(split='train',
              is_gold_summaries=False,
              data_file_path = "/Users/secilsen/PycharmProjects/LongDocumentSummarization/data"):
    """
    Load the dataset from a csv file, if the file does not exist load it from huggingface and save it in json format.

    :param split: train, valid or test
    :param is_gold_summaries: specifies whether the abstracts should be loaded
    :param data_file_path: file in csv format that contains the dataset
    :return: the dataset as pandas dataframe
    """
    data_file_path = f"{data_file_path}/{split}.json"
    print("Data loading....")

    if os.path.exists(data_file_path):
        print("Data loading ends.")
        return load_data_from_file(data_file_path, is_gold_summaries)

    dataset = load_dataset("ccdv/arxiv-summarization", split=split)
    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.sample(frac=0.1)
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

def load_data_from_file_as_df(data_file_path):
    df = pd.read_csv(data_file_path)
    return df


def load_data_as_df(split='train',
                    data_file_path = "/Users/secilsen/PycharmProjects/LongDocumentSummarization/data"):
    data_file_path = f"{data_file_path}/{split}.csv"
    print("Data loading....")

    if os.path.exists(data_file_path):
        return load_data_from_file_as_df(data_file_path)

    dataset = load_dataset("ccdv/arxiv-summarization", split=split)
    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.sample(frac=0.1)
    df_dataset["article"] = df_dataset["article"].apply(lambda x: cleanup(x))
    df_dataset["abstract"] = df_dataset["abstract"].apply(lambda x: cleanup(x))
    df_dataset["topic"] = df_dataset["article"].apply(lambda x: BertTopicForSummarization()())

    return df_dataset


## Util methods for BERTopic visualisation

def print_topic_size(topic_model: BERTopic, docs):
    """
    Visualize BERTopic

    :param topic_model: a trained BERTopic model to visualize
    :param docs: documents to analyze
    """
    print(topic_model.topics)
    topic_model.visualize_barchart()
    new_topics = topic_model.reduce_topics(docs, topic_model.topics)
    print(new_topics)












