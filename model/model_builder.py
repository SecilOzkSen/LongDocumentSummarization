import torch.nn as nn
import torch
from transformers import LongformerModel,LongformerConfig

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