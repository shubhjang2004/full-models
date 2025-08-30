import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import BartLearnedPositionalEmbedding,Modelargs
from encoder import BartAttention
from transformers.activations import GELUActivation


class BartDecoder(nn.Module):
    def __init__(self,args:Modelargs,token_embed:nn.Module):
        super().__init__()

        self.embed_tokens=token_embed
        self.embed_positions=BartLearnedPositionalEmbedding(args.seqlen,args.dim)

        self.layers=nn.ModuleList([BartDecoderLayer(args) for _ in range(args.n_layers)])

        self.layernorm_embedding=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)

    
    def forward(self,decoder_input,encoder_output,decoder_padding_mask,causal_mask):

        B,T=decoder_input.size()
        embed=self.embed_tokens(x)
        posn=torch.arange(0,T,dtype=torch.long,device=x.device)
        posn_embed=self.embed_positions(posn)
        x=embed+posn_embed

        for layer in self.layers:
            x=layer(x,encoder_output,decoder_padding_mask,causal_mask)

        decoder_output=self.layernorm_embedding(x)
        return decoder_output    


class BartDecoderLayer(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.self_attn=BartAttention(args)
        self.activation_dn=GELUActivation()
        self.self_attn_layer_norm=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)
        self.encoder_attn=BartAttention(args)
        self.encoder_attn_layer_norm=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)
        self.fc1=nn.Linear(args.dim,4*args.dim)
        self.fc2=nn.Linear(4*args.dim,args.dim)
        self.final_layer_norm=nn.LayerNorm(args.dim,eps=1e-5,elementwise_affine=True)


    def forward(self,x,encoder_output,padding_mask,causal_mask):

        x=x + self.self_attn_layer_norm(self.self_attn(x,padding_mask,causal_mask))

        x=x + self.encoder_attn_layer_norm(self.encoder_attn(x,encoder_output,padding_mask))

        x=x + self.final_layer_norm(self.fc2(self.activation(self.fc1(x))))

        return x


        
