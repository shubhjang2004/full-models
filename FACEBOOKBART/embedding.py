import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class Modelargs:
    dim:int=768
    vocab_size:int=50265
    seqlen:int=1024
    n_heads:int=12
    n_layers:int=6


class BartScaledWordEmbedding(nn.Module):
    def __init__(self,vocab_size,embed,padding_idx=1):
        super().__init__()
        
        self.embedding=nn.Embedding(vocab_size,embed,padding_idx)

    def forward(self,x):
        return self.embedding(x) 


class BartLearnedPositionalEmbedding(nn.Module):
    def __init__(self,seqlen,embed):
        super().__init__()
        self.posn_embedding=nn.Embedding(seqlen,embed)

    def forward(self,x):
        return self.posn_embedding(x)           

