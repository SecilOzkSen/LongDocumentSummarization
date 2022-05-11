from datasets import load_dataset
import os
import json
import re
from constants import (TRAIN_FILE_PATH,
                       VALIDATION_FILE_PATH,
                       TEST_FILE_PATH)

def load_huggingface_dataset_from_hub():
    dataset = load_dataset("scientific_papers", 'arxiv')
    train_df = dataset["train"].to_pandas()
    validation_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df = clean_latex_symbols_from_string(train_df)
    validation_df = clean_latex_symbols_from_string(validation_df)
    test_df = clean_latex_symbols_from_string(test_df)

    train_df.to_json(TRAIN_FILE_PATH)
    validation_df.to_json(VALIDATION_FILE_PATH)
    test_df.to_json(TEST_FILE_PATH)

    return load_from_json()


def load_from_json():
    #TODO - we need to clean the dataset from undefined chars.
    with open(TRAIN_FILE_PATH, 'r') as fp:
        train_dataset = json.load(fp)
    with open(VALIDATION_FILE_PATH, 'r') as fp:
        validation_dataset = json.load(fp)
    with open(TEST_FILE_PATH, 'r') as fp:
        test_dataset = json.load(fp)
    return train_dataset, validation_dataset, test_dataset


def load_data():
    if os.path.exists(TRAIN_FILE_PATH) == False or \
            os.path.exists(VALIDATION_FILE_PATH) == False or os.path.exists(TEST_FILE_PATH) == False:
        return load_huggingface_dataset_from_hub()
    else:
        return load_from_json()

def clean_latex_symbols_from_string(dataframe):
    regex_for_latex = r"(\${1,2})(?:(?!\1)[\s\S])*\1"
    dataframe["abstract"].apply(lambda x: re.sub(regex_for_latex, "", x))
    dataframe["article"].apply(lambda x: re.sub(regex_for_latex, "", x))
    return dataframe








