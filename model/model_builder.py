import torch.nn as nn
import torch
from transformers import LongformerModel,LongformerConfig
from encoder import TransformerInterEncoder
from torch.nn.init import xavier_uniform_

class Longformer(nn.Module):
    def __init__(self, args):
        super(Longformer, self).__init__()

        config = LongformerConfig.from_pretrained('allenai/'+args.base_LM)
        config.attention_window=args.local_attention_window

        self.model = LongformerModel.from_pretrained('allenai/'+args.base_LM, cache_dir=args.temp_dir,config=config)
        self.finetune = args.finetune_bert
        self.use_global_attention = args.use_global_attention

    def forward(self, x, mask, clss):
        #position_ids
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)


        #attention_mask
        attention_mask = mask.long()


        #global_attention_mask
        global_attention_mask = torch.zeros(x.shape, dtype=torch.long, device=x.device)
        global_attention_mask[:, clss] = 1


        if(self.finetune):

            if (self.use_global_attention):
                top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).last_hidden_state
            else:
                top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=None).last_hidden_state

        else:

            self.eval()
            with torch.no_grad():

                if (self.use_global_attention):
                    top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).last_hidden_state
                else:
                    top_vec  = self.model(x, position_ids=position_ids, attention_mask=attention_mask, global_attention_mask=None).last_hidden_state
        return top_vec

class LongDocumentSummarizer(nn.Module):
    def __init__(self,
                 args,
                 device):
        super(LongDocumentSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.longformer = Longformer(args.temp_dir)
        self.encoder = TransformerInterEncoder(self.longformer.model.config.hidden_size, args.ff_size, args.heads,
                                               args.dropout, args.inter_layers)
        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls