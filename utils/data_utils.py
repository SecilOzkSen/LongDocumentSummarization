from datasets import load_dataset
import os
import json
from constants import (TRAIN_FILE_PATH,
                       VALIDATION_FILE_PATH,
                       TEST_FILE_PATH)

def load_huggingface_dataset_from_hub():
    dataset = load_dataset("scientific_papers", 'arxiv')
    dataset["train"].to_json(TRAIN_FILE_PATH)
    dataset["validation"].to_json(VALIDATION_FILE_PATH)
    dataset["test"].to_json(TEST_FILE_PATH)
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


load_huggingface_dataset_from_hub()







