from data_operations.datataset_wrapper import ArxivSummaryWithTopicDataModule
from utils.data_utils import load_data_v2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.summarizer_model import AbstractiveLongDocumentSummarizerModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

MODEL_NAME_OR_PATH = 'allenai/led-large-16384'
N_EPOCHS = 5

df_train = load_data_v2('train')[0:2]
df_validation = load_data_v2('validation')[0:2]
df_test = load_data_v2('test')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

data_module = ArxivSummaryWithTopicDataModule(train_df=df_train,
                                              validation_df=df_validation,
                                              test_df=df_test,
                                              tokenizer=tokenizer,
                                              model=model)
model = AbstractiveLongDocumentSummarizerModel()

checkpoint_callback = ModelCheckpoint(dirpath="/Users/secilsen/PycharmProjects/LongDocumentSummarization/checkpoints",
                                      filename="summarizer-checkpoint",
                                      save_top_k=1,
                                      verbose=True,
                                      monitor="val_loss",
                                      mode="min")

logger = TensorBoardLogger("/Users/secilsen/PycharmProjects/LongDocumentSummarization/lightning_logs",
                           name="longformer-with-topic-summarizer",
                           )
trainer = pl.Trainer(logger=logger,
                   checkpoint_callback=checkpoint_callback,
                   max_epochs=N_EPOCHS,
                   progress_bar_refresh_rate=30
                   )

if __name__ == "__main__":
    trainer.fit(model, data_module)