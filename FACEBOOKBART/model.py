import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import BartEncoder
from decoder import BartDecoder

from embedding import BartScaledWordEmbedding,Modelargs






class BartForConditionalGeneration(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.model=BartModel(args)
        self.lm_head=nn.Linear(args.dim,args.vocab_size)

        
        self.register_buffer("causal_mask",torch.tril(torch.ones(args.seqlen,args.seqlen)).view(
                                                  1,1,args.seqlen,args.seqlen ))
        
                                                            


    def forward(self,encoder_input,
                encoder_padding_mask,
                decoder_input,
                decoder_padding_mask,
                target=None):
        
        decoder_output=self.model(encoder_input,encoder_padding_mask,
                               decoder_input,decoder_padding_mask,self.causal_mask)    
        
        logits=self.lm_head(decoder_output)

        if target is not None:
             loss=nn.CrossEntropyLoss(logits.view(-1,logits.size(-1)),target.view(-1),
                                      ignore_index=-100,reduction="mean")
             return logits,loss
        
        loss=None
        logits,loss

   




class BartModel(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.shared=BartScaledWordEmbedding(args.vocab_size,args.dim,padding_idx=1)

        self.encoder=BartEncoder(args,self.shared)
        self.decoder=BartDecoder(args,self.shared)

    def forward(self,encoder_input,encoder_padding_mask,decoder_input,decoder_padding_mask,causal_mask):
          encoder_output=self.encoder(encoder_input,encoder_padding_mask)
          decoder_output=self.decoder(decoder_input,encoder_output,decoder_padding_mask,causal_mask)

          return decoder_output 






args=Modelargs()
model=BartForConditionalGeneration(args)
state_dict=model.state_dict()
sd_keys=state_dict.keys()

sd_keys=[k for k in sd_keys if not k.endswith("causal_mask")]
from transformers import BartForConditionalGeneration,BartTokenizer
model2= BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer=BartTokenizer.from_pretrained("facebook/bart-base")
sd_hf=model2.state_dict()   

"""
for k in sd_keys:
    if((state_dict[k].shape==sd_hf[k].shape) ):
        print(f"{k} matched")

    else:
        print(f"{k} didn't matched")     
"""
print(sd_keys)        