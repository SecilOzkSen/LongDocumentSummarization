from bert_topic import BertTopicForSummarization
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from enum import Enum, auto
from typing import List

class PoolingOp(Enum):
    max = auto()
    average = auto()

    def equals(self, string: str) -> bool:
        return self.name == string

class TopicEmbeddingExtractor():

    def __init__(self,
                 embedding_model_name_or_path: str = "bert-base-uncased",
                 pooling_op: str = "average"):
        if pooling_op not in (PoolingOp.max.name, PoolingOp.average.name):
            raise ValueError(f"Unsupported pooling operation ({pooling_op})")
        self._model = AutoModelWithLMHead.from_pretrained(embedding_model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(embedding_model_name_or_path)
        self._pooling_op = pooling_op
        self.bert_topic = BertTopicForSummarization()

    def produce_embeddings(self, span:str):
        with torch.no_grad():
            encoded_span = self._tokenizer.encode(span, add_special_tokens=True)
            input_ids = torch.tensor([encoded_span])
            last_hidden_states = self._model(input_ids)[0]
        return last_hidden_states[:, 1:-1, :]

    def __call__(self, doc:str, is_pooling: bool = False) -> List[float]:
        topic = self.bert_topic(doc) #TODO: bert topic needs to give topics for given document.
        embedding = self.produce_embeddings(topic)
        if is_pooling:
            pool_op_dim = embedding.size()
            if PoolingOp.max.equals(self._pooling_op):
                pool_op = torch.nn.AdaptiveMaxPool2d(pool_op_dim[1], pool_op_dim[2])
            else:
                pool_op = torch.nn.AdaptiveAvgPool2d(pool_op_dim[1], pool_op_dim[2])
            return pool_op(embedding)
        return embedding
