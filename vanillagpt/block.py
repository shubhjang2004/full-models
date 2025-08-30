import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.activations import NewGELUActivation
from dataclasses import dataclass

@dataclass
class Config:
    dim:int=768
    hidden_dim:int=3072
    n_heads:int=12
    seqlen:int=2048
    vocab_size:int=50257
    n_layers:int=12
    dropout:float=0.0
    


class GPTNeoBlock(nn.Module):
    def __init__(self,args:Config):
        super().__init__()
        self.ln_1=nn.LayerNorm(args.dim)
        self.attn=GPTNeoattention(args)
        self.ln_2=nn.LayerNorm(args.dim)
        self.mlp=GPTNeoMLP(args)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    



class GPTNeoselfattention(nn.Module):
    def __init__(self,args:Config):
        super().__init__()
        self.dim=args.dim
        self.n_heads=args.n_heads
        self.head_dim=args.dim//args.n_heads
        self.attn_dropout=nn.Dropout(args.dropout,inplace=False)
        self.resd_dropout=nn.Dropout(args.dropout,inplace=False)
        self.k_proj=nn.Linear(self.dim,self.dim,bias=False)
        self.v_proj=nn.Linear(self.dim,self.dim,bias=False)
        self.q_proj=nn.Linear(self.dim,self.dim,bias=False)
        self.out_proj=nn.Linear(self.dim,self.dim,bias=True)

        


    def forward(self,x):
        B,T,C=x.size()
        assert C == self.dim and C%self.n_heads==0
        key=self.k_proj(x)
        value=self.v_proj(x)
        query=self.q_proj(x)
        key=key.view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        value=value.view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        query=query.view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
        
        attn=query@key.transpose(-1,-2)/math.sqrt(self.head_dim)
        attn=attn.masked_fill(mask[:,:,:T,:T]==0,float('-inf'))
        attn=F.softmax(attn,dim=-1)
        attn=self.attn_dropout(attn)
        y=attn@value
        y=y.transpose(1,2).contiguous().view(B,T,C)

        return self.resd_dropout(self.out_proj(y))

        
 
class GPTNeoattention(nn.Module):
    def __init__(self,args:Config):
        super().__init__()
        self.attention=GPTNeoselfattention(args)

    def forward(self,x):
        return self.attention(x)
    


class GPTNeoMLP(nn.Module):
    def __init__(self,args:Config):
        super().__init__()
        self.c_fc=nn.Linear(args.dim,args.hidden_dim,bias=True)
        self.c_proj=nn.Linear(args.hidden_dim,args.dim)
        self.act=NewGELUActivation()

        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x):
        x=self.c_proj(self.act(self.c_fc(x)))
        x=self.dropout(x)     
        return x   
""""
class Layernorm(nn.Module):
    def __init__(self,dim:int,eps:float,elementwise_affine=True):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.bias=nn.Parameter(torch.empty(self.dim))
        self.weight=nn.Parameter(torch.empty(self.dim))
        torch.nn.init.normal_(self.weight,mean=0.0,std=0.02)
        torch.nn.init.zeros_(self.bias)

    def forward(self,x):

        x_mean=torch.mean(x,dim=-1,keepdim=True)
        x_std=torch.std(x,dim=-1,keepdim=True)
        x_norm=self.weight*(x-x_mean)/(x_std+self.eps)
        return x_norm+self.bias 



"""
"""
class Layernorm(nn.Module):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)  # population variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm

"""            