import numpy as np
import pandas as pd
from transformers import LongformerTokenizer, LongformerModel
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from typing import Optional
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from torch.nn import TransformerEncoderLayer

class ArxivSummaryWithTopicDataModule(LightningDataModule):
    """
    Dataset handler, stores the train, validation and testset. The data is passed as pandas dataframes.
    It makes use of the ArxivSummaryWithTopicDataset to convert the documents to the input representation.
    The final input representation is build using Longformer sentence embeddings and embedding BERTopic
    and passing this representation through a transformer layer. The output contains the corresponding representation
    of the document and summary and the respective attention masks.
    """
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
    """
    One dataset of documents that are converted into the input representation which combines longformer sentence embeddings and topic embeddings.
    """
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
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)


    def __len__(self):
        return len(self.data)

    def _produce_embeddings(self, pretokenized_span:str):
        """
        Get the Embedding of a span from the Longformer tokenizer.

        :param pretokenized_span:
        :return:
        """
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


    def prepare_input_embedding(self, doc, topic):
        """
        Get the input embeddings of a document by adding the longfomer sentence embedding and the topic embedding for each sentence.

        :param doc: the document to be represented
        :param topic: str, the topic classification by BERTopic
        :return: the document embedding as a numpy array
        """
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
        """
        Get the input representation of one document and the representation of the gold summary.
        :param index: id of the target document
        :return: dict: text: of the document,
                topic: predicted by BERTopic,
                summary: the abstract of the article,
                text_input_ids: new input representation from the transformer a conversion of the sentence and topic embedding,
                text_attention_mask: the attention mask from the transformer layer,
                labels: representation of the gold summary,
                labels_attention_mask: the corresponding attention mask
        """
        data_row = self.data.iloc[index]

        text = data_row['article']
        topic = data_row['topic']
        summary = data_row['abstract']

        summed_embedding = self.prepare_input_embedding(text, topic)
        #Transformer layer for learning the new input representation (as in T-BertSum (see t-bertsum embedding fig.))
        summed_encoding = self.transformer_encoder_layer(summed_embedding)

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_token_limit,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        labels=summary_encoding["input_ids"]
        labels[labels==0] = -100 # change padding zero tokens with -100.

        return dict(
            text=text,
            topic=topic,
            summary=summary,
            text_input_ids=summed_encoding["input_ids"].flatten(),
            text_attention_mask=summed_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten(),
        )








