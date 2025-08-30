import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUActivation
from dataclasses import dataclass
from embedding import BartLearnedPositionalEmbedding,Modelargs




class BartEncoder(nn.Module):
    def __init__(self,args:Modelargs,token_embed:nn.Module):
        super().__init__()
        self.embed_tokens=token_embed
        self.embed_positions=BartLearnedPositionalEmbedding(1026,768)

        self.layers=nn.ModuleList(BartEncoderLayer(args) for _ in range(args.n_layers))
        self.layernorm_embedding=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)


    def forward(self,encoder_input,encoder_padding_mask):
        B,T=encoder_input.size() 
        embed=self.embed_tokens(encoder_input)
        posn=torch.arange(0,T,dtype=torch.long)
        posn_embed=self.embed_positions(posn)
        x=embed+posn_embed
        for layer in self.layers:
            x=layer(x,encoder_padding_mask)
        encoder_output=self.layernorm_embedding(x)
        return encoder_output 


        


class BartEncoderLayer(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.self_attn=BartAttention(args)
        self.self_attn_layer_norm=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)
        self.activation_fn=GELUActivation()
        self.fc1=nn.Linear(args.dim,4*args.dim)
        self.fc2=nn.Linear(4*args.dim,args.dim)
        self.final_layer_norm=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)

    def forward(self,x,padding_mask):
        x=x+self.self_attn_layer_norm(self.self_attn(x,padding_mask))
        encoder_output=x+self.final_layer_norm(self.fc2(self.activation_fn(self.fc1)))
        
        return encoder_output



class BartAttention(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.k_proj=nn.Linear(args.dim,args.dim)
        self.v_proj=nn.Linear(args.dim,args.dim)
        self.q_proj=nn.Linear(args.dim,args.dim)

        self.out_proj=nn.Linear(args.dim,args.dim)
        self.n_heads=args.n_heads
        self.head_dim=args.dim//args.n_heads

        self.scale_factor=self.head_dim**-0.5

    def forward(self,x,encoder_output=None,padding_mask=None,causal_mask=None):
        B,T,C=x.size()
        q=self.q_proj(x)
        if encoder_output is not None:
            k=self.k_proj(encoder_output)
            v=self.v_proj(encoder_output)
        else:
            k=self.k_proj(x)
            v=self.v_proj(x)
        

        query=q.view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        key=k.view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        value=v.view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)

        attn=query@key.transpose(-2,-1)*self.scale_factor

        if padding_mask is not None:
            attn=attn.masked.fill(padding_mask[:,:,:T,:T]==0,float("-inf"))
        
        if causal_mask is not None:
            attn=attn.masked.fill(causal_mask[:,:,:T,:T]==0,float('-inf'))

        attn=F.softmax(attn,dim=-1)

        y=attn@value
        y = y.transpose(1,2).view(B,T,C)
        y=self.out_proj(y)
        return y
        

        


