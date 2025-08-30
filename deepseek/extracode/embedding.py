import torch
import torch.nn as nn
import torch.nn.functional as F


"""
class Embedding(nn.Module):
    def __init__(self,vocab_size:int,dim:int):
        super().__init__()
        self.vocab_size=vocab_size
        self.dim=dim
        self.weight=nn.Parameter(torch.empty(self.vocab_size,self.dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self,x):
        return F.embedding(x,self.weight)    
    

"""

### IF YOU HAVE spread embedding accross multiple gpus
""""
world_size=1
rank=0
class ParallelEmbedding(nn.Module):
    def __init__(self,dim:int,vocab_size:int):
        super().__init__()
        self.dim=dim
        self.vocab_size=vocab_size
        if world_size>1:
            assert vocab_size%world_size==0
        self.local_vocab_size=vocab_size/world_size
        self.local_start_idx=rank*self.local_vocab_size
        self.local_end_idx=self.local_start_idx + self.local_vocab_size
        self.weight=nn.Parameter(torch.empty(self.local_vocab_size,self.dim))


    def forward(self,x):
        if world_size>1:
            mask=x<self.local_start_idx|x>self.local_end_idx
            x=x-self.local_start_idx
            x[mask]=0
        y=F.embedding(self.weight,x)
       
        if(world_size>1):
            y[mask]=0;  
            dist.all_reduce(y)  
        return y

"""
