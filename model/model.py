from transformers import AutoModel, AutoTokenizer
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from datasets import load_metric
from rouge import Rouge

rouge = load_metric('rouge')
rouge_v2 = Rouge()

class AbstractiveLongDocumentSummarizerModel(LightningModule):
    """
    The document summarization model which uses the pretrained Longfromer model.
    """
    def __init__(self,
                 model_name_or_path = 'allenai/led-large-16384',
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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

    def compute_metrics(self, pred, rouge_type):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=[rouge_type]
        )[rouge_type].mid

        return {
            f"{rouge_type}_precision": round(rouge_output.precision, 4),
            f"{rouge_type}_recall": round(rouge_output.recall, 4),
            f"{rouge_type}_f1": round(rouge_output.fmeasure, 4),
            }


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

        scores = rouge_v2.get_scores(outputs, batch["summary"])
        rouge1 = scores['rouge-1']
        rouge2 = scores['rouge-2']
        rougeN = scores['rouge-l']

        self.log("rouge_1", rouge1['f'], prog_bar=True, logger=True)
        self.log("rouge_2", rouge2['f'], prog_bar=True, logger=True)
        self.log("rouge_n", rougeN['f'], prog_bar=True, logger=True)

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
        return AdamW(self.parameters(), lr=0.0001)

