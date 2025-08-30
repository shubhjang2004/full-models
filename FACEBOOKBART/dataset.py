from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import BartForConditionalGeneration,BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer=BartTokenizer.from_pretrained("facebook/bart-base")

from datasets import load_dataset
ds = load_dataset("abisee/cnn_dailymail", "1.0.0")




text_tokens=[]
summary_tokens=[]
for i,example in enumerate(ds["train"]):
    text=example["article"]
    text=tokenizer.encode(text)
    text_tokens.append(text)
    summary=example["highlights"]
    summary=tokenizer.encode(summary)
    summary_tokens.append(summary)
    if(i==20000):
        break


bos_token_id=tokenizer.bos_token_id
eos_token_id=tokenizer.eos_token_id
pad_token_id=tokenizer.pad_token_id
max_src_len=1024
max_tgt_len=128

encoder_inputs=[]
encoder_attn_mask=[]
decoder_inputs=[]
decoder_attn_mask=[]
decoder_target=[]

for src_ids,tgt_ids in zip(text_tokens,summary_tokens):
    if(len(src_ids)>=max_src_len):
        enc_in=src_ids[:max_src_len-1]
        enc_in=enc_in+[eos_token_id]
    else:
       enc_in=src_ids+[eos_token_id]
       src_pad_len=max_src_len-len(enc_in)
       enc_in=enc_in+[pad_token_id]*src_pad_len

    enc_attn_mask=[1 if token != pad_token_id else 0 for token in enc_in ]

    if(len(tgt_ids)>=max_tgt_len):
        tgt_ids=tgt_ids[:max_tgt_len-1]
        dec_in=[bos_token_id]+tgt_ids
        target=tgt_ids+[eos_token_id]

    else:
        dec_in=[bos_token_id]+tgt_ids
        target=tgt_ids+[eos_token_id]

        dec_pad_len=max_tgt_len-len(dec_in)
        dec_in=dec_in+[pad_token_id]*dec_pad_len
        target=target+[pad_token_id]*dec_pad_len

    dec_attn_mask=[1 if token !=pad_token_id else 0 for token in dec_in]      
    target=[token if token!= pad_token_id else -100 for token in target]
    encoder_inputs.append(enc_in)
    encoder_attn_mask.append(enc_attn_mask)
    decoder_inputs.append(dec_in)
    decoder_attn_mask.append((dec_attn_mask))
    decoder_target.append(target)       


class SumarizationDataset(Dataset):
    def __init__(self,encoder_inputs,encoder_attn_mask,decoder_inputs,decoder_attn_mask,decoder_target):
        super().__init__()
        self.encoder_inputs=torch.tensor(encoder_inputs,dtype=torch.long)
        self.encoder_attn_mask=torch.tensor(encoder_attn_mask,dtype=torch.long)
        self.decoder_inputs=torch.tensor(decoder_inputs,dtype=torch.long)
        self.decoder_attn_mask=torch.tensor(decoder_attn_mask,dtype=torch.long)
        self.decoder_target=torch.tensor(decoder_inputs,dtype=torch.long)

    def __len__(self):
        return len(self.encoder_inputs)
    
    def __getitem__(self,idx):
        return{
            "encoder_inputs":self.encoder_inputs[idx],
            "encoder_attn_mask":self.encoder_attn_mask[idx],
            "decoder_inputs":self.decoder_inputs[idx],
            "decoder_attn_mask":self.decoder_attn_mask[idx],
            "target":self.decoder_target[idx]

        }

train_dataset=SumarizationDataset(encoder_inputs,
                                  encoder_attn_mask,
                                  decoder_inputs,
                                  decoder_attn_mask,
                                  decoder_target)

train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)


