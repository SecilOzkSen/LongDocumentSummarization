import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow_datasets.summarization.scientific_papers import ScientificPapers

if __name__ == "__main__":

    #pip install datasets
from datasets import load_dataset

dataset = load_dataset("ccdv/arxiv-summarization")
print(dataset)

with open("sum_data.pkl", "wb" ) as p:
    pickle.dump(dataset, p)
"""
if __name__ == "__main__":
    builder = tfds.builder('scientific_papers')
    builder.download_and_prepare()
# 2. Load the `tf.data.Dataset`
    ds = builder.as_dataset(split='train', shuffle_files=True)
    print(ds)
 #   ds = tfds.load('scientific_papers', split='train', shuffle_files=True)
   # arxiv = ScientificPapers()
   # dataset_dataframe = tfds.as_dataframe()
"""





