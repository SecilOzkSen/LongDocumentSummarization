from transformers import LongformerTokenizer, LongformerModel
from pytorch_lightning import LightningModule
from torch.optim import AdamW

class AbstractiveLongDocumentSummarizerModel(LightningModule):

    def __init__(self,
                 model_name_or_path = 'allenai/led-large-16384',
                 ):
        super().__init__()
        self.model = LongformerModel.from_pretrained(model_name_or_path, return_dict=True)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self.forward(input_ids=input_ids,
             attention_mask=attention_mask,
             labels=labels,
             decoder_attention_mask=labels_attention_mask)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self.forward(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels,
                                     decoder_attention_mask=labels_attention_mask)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self.forward(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels,
                                     decoder_attention_mask=labels_attention_mask)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001, )

