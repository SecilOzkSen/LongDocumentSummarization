import numpy as np
import pandas as pd
from transformers import LongformerTokenizer, LongformerModel
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from typing import Optional
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

class ArxivSummaryWithTopicDataModule(LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 validation_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 tokenizer: LongformerTokenizer,
                 batch_size: int = 8,
                 text_max_token_limit :int = 8192,
                 summary_max_token_length : int = 512):
        super().__init__()
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.text_max_token_limit = text_max_token_limit
        self.summary_max_token_limit = summary_max_token_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArxivSummaryWithTopicDataset(
            data=self.train_df,
            tokenizer=self.tokenizer,
            input_token_limit=self.text_max_token_limit,
            summary_token_limit=self.summary_max_token_limit
        )
        self.validation_dataset = ArxivSummaryWithTopicDataset(
            data=self.validation_df,
            tokenizer=self.tokenizer,
            input_token_limit=self.text_max_token_limit,
            summary_token_limit=self.summary_max_token_limit
        )

        self.test_dataset = ArxivSummaryWithTopicDataset(
            data=self.test_df,
            tokenizer=self.tokenizer,
            input_token_limit=self.text_max_token_limit,
            summary_token_limit=self.summary_max_token_limit
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2)



class ArxivSummaryWithTopicDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: LongformerTokenizer,
                 input_token_limit: int = 8192,
                 summary_token_limit: int = 512):
        self.tokenizer = tokenizer
        self.data = data
        self.input_token_limit = input_token_limit
        self.summary_token_limit = summary_token_limit
        self.sentence_transformer_model = SentenceTransformer('allenai/longformer-base-4096')

    def __len__(self):
        return len(self.data)

  #  def _prepare_input_document(self, doc):
  #      sentences = doc.split('\n')
  #      new_doc = '[CLS] '
  #      for sentence in sentences:
  #          new_doc = new_doc + sentence + ' [SEP] '

   #     return new_doc.rstrip(), len(sentences)

    def _produce_embeddings(self, pretokenized_span:str):
        with torch.no_grad():
            encoded_span = self.tokenizer.encode(pretokenized_span)
            input_ids = torch.tensor([encoded_span])
            last_hidden_states = self._model(input_ids)[0]
        return last_hidden_states[:, 1:-1, :]

    def _input_document_sentence_length(self, doc):
        return len(doc.split('\n'))

    def _prepare_topic(self, topic, text_sentence_length):
        topic_doc = ''
        for i in range(text_sentence_length):
            topic_doc = topic_doc + topic + '. '
        return topic_doc.rstrip()


  #  def _prepare_topic(self, topic, text_sentence_length):
  #      topic_doc = '[CLS] '
  #      for i in range(text_sentence_length):
  #          topic_doc = topic_doc + topic + ' [SEP] '
  #      return topic_doc.rstrip()

    def prepare_input_embedding(self, doc, topic):
        topic_embedding = self.sentence_transformer_model.encode(topic, convert_to_numpy=True)
        summed_embeddings = []
        for sentence in doc.split('\n'):
            sentence_embedding = self.sentence_transformer_model.encode(sentence, convert_to_numpy=True)
            sum_embedding = sentence_embedding + topic_embedding
            summed_embeddings.append(sum_embedding)
        summed_embeddings = np.array(summed_embeddings)
        summed_embeddings.flatten()
        return summed_embeddings


    def __getitem__(self, index:int):
        data_row = self.data.iloc[index]
        text = data_row['article']
        topic = data_row['topic']
        summary = data_row['abstract']

    #    text_sentence_length = self._input_document_sentence_length(text)
    #    prepared_topic = self._prepare_topic(topic, text_sentence_length)

    #    text_encoding = self.tokenizer.encode(text,
    #                                             padding="max_length",
    #                                             truncation=True,
    #                                             max_length=self.input_token_limit,
    #                                             return_attention_mask = True,
    #                                             return_tensors="pt")
    #    text_embedding = self._produce_embeddings(tokenized_text)

    #    topic_encoding = self.tokenizer.encode(prepared_topic,
    #                                              padding="max_length",
    #                                              truncation=True,
    #                                              max_length=self.input_token_limit)
        summed_embedding = self.prepare_input_embedding(text, topic)
    #    text_input_ids = torch.tensor([text_encoding])
    #    text_last_hidden_states = self._model(text_input_ids)[0]


     #   topic_embedding = self._produce_embeddings(tokenized_topic)

     #   input_embedding = torch.add(text_embedding, topic_embedding)

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_token_limit,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return dict(
            text=text,
            topic=topic,
            summary=summary,
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            topic_input_ids=topic_encoding["input_ids"].flatten(),
            topic_attention_mask=topic_encoding["attention_mask"].flatten(),
            summary_input_ids=summary_encoding["input_ids"].flatten(),
            summary_attention_mask=summary_encoding["attention_mask"].flatten()
        )








