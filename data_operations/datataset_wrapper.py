import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from typing import Optional, List
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from torch.nn import Transformer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
import math
from copy import deepcopy

class ArxivSummaryWithTopicDataModule(LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 validation_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 model: AutoModelForSeq2SeqLM,
                 batch_size: int = 2,
                 text_max_token_limit :int = 8192,
                 summary_max_token_length : int = 500):
        super().__init__()
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.text_max_token_limit = text_max_token_limit
        self.summary_max_token_limit = summary_max_token_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArxivSummaryWithTopicDataset(
            dataframe=self.train_df,
            tokenizer=self.tokenizer,
            model=self.model,
            input_token_limit=self.text_max_token_limit,
            summary_token_limit=self.summary_max_token_limit,
        )
        self.validation_dataset = ArxivSummaryWithTopicDataset(
            dataframe=self.validation_df,
            tokenizer=self.tokenizer,
            model=self.model,
            input_token_limit=self.text_max_token_limit,
            summary_token_limit=self.summary_max_token_limit,
        )

        self.test_dataset = ArxivSummaryWithTopicDataset(
            dataframe=self.test_df,
            tokenizer=self.tokenizer,
            model=self.model,
            input_token_limit=self.text_max_token_limit,
            summary_token_limit=self.summary_max_token_limit,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)



class ArxivSummaryWithTopicDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 model: AutoModelForSeq2SeqLM,
                 input_token_limit: int = 8192,
                 summary_token_limit: int = 500):
        self.tokenizer = tokenizer
        self.model = model
        self.dataset = dataframe
        self.input_token_limit = input_token_limit
        self.summary_token_limit = summary_token_limit
        self.transformer_layer = Transformer(d_model=1024, nhead=8, num_encoder_layers=12)
        self.spacy_model = SpacySentenceSplitter(language='en_core_web_sm')

    def token_based_embedding_producer(self, txt):

        input_ids = torch.tensor(self.tokenizer.encode(txt)).unsqueeze(0)

        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                    device=input_ids.device)  # initialize to local attention
        attention_mask[:, [0, -1]] = 2

        with torch.no_grad():

            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs[2] #last hidden state
            token_embeddings = torch.squeeze(hidden_states, dim=0)

        return token_embeddings


    def token_based_embedding_summing(self, sentence, topic):
        sentence_token_embeddings = self.token_based_embedding_producer(sentence)
        topic_token_embeddings = self.token_based_embedding_producer(topic)
        sentence_token_len = sentence_token_embeddings.shape[0]
        topic_token_len = topic_token_embeddings.shape[0]
        sentence_token_embeddings = sentence_token_embeddings.numpy()
        topic_token_embeddings = topic_token_embeddings.numpy()

        n_reproduce = math.floor(sentence_token_len / topic_token_len)
        residual = sentence_token_len - (n_reproduce*topic_token_len)

        new_topic_embedding = topic_token_embeddings

        for i in range(n_reproduce-1):
            new_topic_embedding = np.concatenate((new_topic_embedding, topic_token_embeddings), axis=0)

        if residual > 0:
            for i in range(residual):
                token_to_concat = topic_token_embeddings[i][:]
                token_to_concat = np.expand_dims(token_to_concat, axis=0)
                new_topic_embedding = np.concatenate((new_topic_embedding, token_to_concat), axis=0)

        sum = sentence_token_embeddings + new_topic_embedding
        #Max pool
        return sum

    def __len__(self):
        return len(self.dataset)

    def pad_input(self,  embedding, pad_value):
        padding_len = self.input_token_limit - embedding.shape[0]
        if padding_len < 0:
            return embedding[0:self.input_token_limit, :]
        embedding_size = embedding.shape[1]
        pad_array = np.ones(shape=(padding_len, embedding_size), dtype=np.float)*pad_value
        padded_embedding = np.concatenate((embedding, pad_array), axis=0)
        return padded_embedding

    def prepare_input_embedding(self, doc, topic):

        _sentences = self.spacy_model.split_sentences(doc)

        def filter_sentences():
            list_sentences = [sentence.replace('\n', '') for sentence in _sentences]  # Data filtering again.
            _list_sentences = []
            for sentence in list_sentences:
                if len(sentence.split()) <= 2:  # Data filtering again.
                    continue
                _list_sentences.append(sentence)
            return _list_sentences

        _list_sentences = filter_sentences()

        summed_embeddings = []
        for i,sentence in enumerate(_list_sentences):
            sum_embedding = self.token_based_embedding_summing(sentence=sentence, topic=topic)
            if i == 0:
                summed_embeddings = sum_embedding
            else:
                summed_embeddings = np.concatenate((summed_embeddings, sum_embedding), axis=0)

    #    pad_token_embedding = self.token_based_embedding_producer('<pad>')
    #    summed_embeddings = self.pad_input(summed_embeddings,
    #                                       pad_value=pad_token_embedding)
        summed_embeddings = torch.Tensor(summed_embeddings)
        summed_embeddings = torch.unsqueeze(summed_embeddings, dim=0)
        summed_embeddings = summed_embeddings.permute(1,0,2)
        return summed_embeddings

    def __getitem__(self, index:int):
        data_row = self.dataset.iloc[index]

        text = data_row['article']
        topic = data_row['topic']
        summary = data_row['abstract']

        summed_embedding = self.prepare_input_embedding(text, topic)
        summary_embedding = self.token_based_embedding_producer(summary)
        summary_embedding = torch.unsqueeze(summary_embedding, dim=0)
        summary_embedding = summary_embedding.permute(1, 0, 2)

        #Transformer layer for learning the new input representation (as in T-BertSum (see t-bertsum embedding fig.))
        summed_encoding = self.transformer_layer(summed_embedding, summary_embedding)
        summed_encoding = summed_encoding.permute(1, 0, 2)
        summed_encoding = torch.squeeze(summed_encoding, dim=0)

        text_encoding = self.tokenizer(
            text,
            max_length=self.input_token_limit,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_token_limit,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        labels=summary_encoding["input_ids"]
        labels[labels == 0] = -100  # change padding zero tokens with -100.

        return dict(
            text=text,
            topic=topic,
            summary=summary,
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_input_embeds = summed_encoding,
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten(),
            contextual_embedding=summed_embedding
        )








