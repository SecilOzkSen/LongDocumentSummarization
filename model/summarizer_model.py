from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from datasets import load_metric
from rouge import Rouge

rouge_v2 = Rouge()

class AbstractiveLongDocumentSummarizerModel(LightningModule):
    """
    The document summarization model which uses the pretrained Longfromer model.

    The input representations are computed with the methods in dataset_wrapper.py and passed to this model.
    """
    def __init__(self,
                 model_name_or_path = 'allenai/led-large-16384-arxiv',
                 ):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def forward(self, input_ids, input_attention_mask, contextual_embeddings, labels, decoder_attention_mask, text_input_embeds):
        output = self.model(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            decoder_input_ids=labels,
            inputs_embed=text_input_embeds,
        #    encoder_outputs=contextual_embeddings,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        encoder_embeds = batch["contextual_embedding"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        text_input_embeds = batch["text_input_embeds"]
        loss, outputs = self.forward(input_ids=input_ids,
                                     input_attention_mask=attention_mask,
                                     contextual_embeddings=encoder_embeds,
                                     labels=labels,
                                     decoder_attention_mask=labels_attention_mask,
                                     text_input_embeds=text_input_embeds)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        encoder_embeds = batch["contextual_embedding"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        text_input_embeds = batch["text_input_embeds"]
        loss, outputs = self.forward(input_ids=input_ids,
                                     input_attention_mask=attention_mask,
                                     contextual_embeddings=encoder_embeds,
                                     labels=labels,
                                     decoder_attention_mask=labels_attention_mask,
                                     text_input_embeds=text_input_embeds)
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
        encoder_embeds = batch["contextual_embedding"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        text_input_embeds = batch["text_input_embeds"]
        loss, outputs = self.forward(input_ids=input_ids,
                                     input_attention_mask=attention_mask,
                                     contextual_embeddings=encoder_embeds,
                                     labels=labels,
                                     decoder_attention_mask=labels_attention_mask,
                                     text_input_embeds=text_input_embeds)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)

