import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



@dataclass
class T5args:
    dim:int=512
    seqlen:int=1024
    vocab_size:int=32128
    n_heads:int=8
    n_layers:int=6
    num_buckets:int=32
    dropout:float=0.1

class T5Out(nn.Module):
    def __init__(self,args:T5args):
        super().__init__()
        self.shared=nn.Embedding(args.vocab_size,args.dim)
        self.encoder=T5Stack(args,self.shared,is_decoder=False)
        self.decode=T5Stack(args,self.shared,is_decoder=True)
        self.lm_head=nn.Linear(args.dim,args.vocab_size,bias=False)


    def forward(self,encoder_id,encoder_mask,decoder_id,decoder_mask,target=None):
        encoder_output=self.encoder(encoder_id,encoder_mask)
        decoder_output=self.decoder(decoder_id,decoder_mask,encoder_output)

        if target is not None:
            logits=self.lm_head(decoder_output)
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        else:
            logits=self.lm_head(decoder_output)
            loss=None

        return logits ,loss    
        
       


class T5Stack(nn.Module):
    def __init__(self,args:T5args,token_embed:nn.Module,is_decoder:bool):
        super().__init__()
        self.embed_tokens=token_embed
        self.block=nn.ModuleList([T5Block(args,is_decoder,is_relative_attention_bias=True) if  i==0 else T5Block(args,is_decoder,is_relative_attention_bias=False)  for i  in range(args.n_layers)])
        self.final_layer_norm=T5LayerNorm(args.dim)
        self.dropout=nn.Dropout(args.dropout)


    def forward(self,x,attn_mask,encoder_output=None):
        x=self.embed_tokens(x)
        if encoder_output:
            for block in self.block:
                x=block(x,attn_mask,encoder_output)
        else:
            for block in self.block:
                x=block(x)

        return x                







class T5Block(nn.Module):
    def __init__(self,args:T5args,is_decoder:bool,is_relative_attention_bias:bool=False):
        super().__init__()
        if is_decoder:
            self.layer=nn.ModuleList([T5LayerSelfAttention(args,is_relative_attention_bias,is_causal=True),T5LayerCrossAttention(args),T5LayerFF(args)])
        else:
            self.layer=nn.ModuleList([T5LayerSelfAttention(args,is_relative_attention_bias,is_causal=False),T5LayerFF(args)])

    def forward(self,x,attn_mask,encoder_output=None):
        if encoder_output:
           for layer in self.layer:
                if layer is T5LayerSelfAttention:
                   x=x+layer(x,attn_mask,encoder_output)
                else:
                   x=x=layer(x,attn_mask)
        else:
            for layer in self.layer:
                x=x+layer(x)

        return x        

                      
            


#-------------Attention------------------------------------------------
class T5LayerCrossAttention(nn.Module):
    def __init__(self,args:T5args):
        super().__init__()
        self.EncDecAttention=T5Attention(args,is_relative_attention_bias=False,is_causal=False)
        self.layer_norm=T5LayerNorm(args.dim)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x,attn_mask,encoder_output):
        x=self.EncDecAttention(x,attn_mask,encoder_output)
        x=self.layer_norm(x)
        x=self.dropout(x)
        return x    



class T5LayerSelfAttention(nn.Module):
    def __init__(self,args:T5args,is_relative_attention_bias:bool,is_causal:bool):
        super().__init__()
        self.SelfAttention=T5Attention(args,is_relative_attention_bias,is_causal)
        self.layer_norm=T5LayerNorm(args.dim)
        self.dropout=nn.Dropout(args.dropout)


    def forward(self,x):
        x=self.SelfAttention(x)
        x=self.layer_norm(x)
        x=self.dropout(x)
 


class T5Attention(nn.Module):
    def __init__(self,args:T5args,is_relative_attention_bias:bool,is_causal:bool):
        super().__init__()
        assert args.dim%args.n_heads==0, f"shape mismatch"
        self.q=nn.Linear(args.dim,args.dim,bias=False)
        self.k=nn.Linear(args.dim,args.dim,bias=False)      
        self.v=nn.Linear(args.dim,args.dim,bias=False)
        self.o=nn.Linear(args.dim,args.dim,bias=False) 

        if is_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(args.num_buckets, args.n_heads)

        if is_causal:
            self.register_buffer("bias",torch.tril(torch.ones(args.seqlen,args.seqlen)).view(1,1,args.seqlen,args.seqlen))    


        self.dim=args.dim
        self.n_heads=args.n_heads
        self.head_dim=args.dim//args.n_heads
        self.scale_factor=self.head_dim**-0.5

        

    def forward(self,x,attn_mask,encoder_output=None):
        B,T,C=x.size()
        query=self.q(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        
        if encoder_output:
            key=self.k(encoder_output).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
            value=self.v(encoder_output).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        else:
            key=self.k(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
            value=self.v(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        

        attn=query@key.transpose(-1,-2)*self.scale_factor
        
        

        num_buckets, num_heads = self.relative_attention_bias.weight.shape
        # Compute relative positions [seqlen, seqlen]
        positions = torch.arange(T, device=attn.device)
        rel_pos = positions[:, None] - positions[None, :]   # [seqlen, seqlen]
        # Bucketize into [0 .. num_buckets-1]
        rel_pos = rel_pos.clamp(-num_buckets // 2, num_buckets // 2 - 1)
        rel_pos = rel_pos + num_buckets // 2   # shift to positive
        bias = self.relative_attention_bias(rel_pos)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias



        mask = attn_mask[:, None, None, :]  # [B, 1, 1, T]
        attn = attn.masked_fill(mask == 0, float('-inf'))

        if self.bias:
            attn=attn.masked_fill(self.bias[:,:,T,:T]==0,float("-inf"))
        attn=F.softmax(attn,dim=-1)

        out=attn@value
        out=out.transpose(1,2).contiguous().view(B,T,C)

        return out  

        





#------------feed forward--------------------------------  

class T5LayerFF(nn.Module):
    def __init__(self,args:T5args):
        super().__init__()
        self.DenseReluDense=T5DenseActDense(args)
        self.layer_norm=T5LayerNorm(args.dim)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x):
        x=self.DenseReluDense(x)
        x=self.layer_norm(x)
        x=self.dropout(x)
        return x    

                 
class T5DenseActDense(nn.Module):
    def __init__(self,args:T5args):
        super().__init__()
        self.wi=nn.Linear(args.dim,4*args.dim,bias=False)
        self.wo=nn.Linear(4*args.dim,args.dim,bias=False)
        self.dropout=nn.Dropout(args.dropout)
        self.act=nn.ReLU()

    def forward(self,x):
        x=self.wi(x)
        x=self.act(x)
        x=self.wo(x)
        x=self.dropout(x)
        return x  
#--------------------------------------------------------------------
#--------------------Layernorm--------------------------------


class T5LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Mean over last dimension
        mean = x.mean(-1, keepdim=True)
       
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (x - mean) / torch.sqrt(variance + 1e-5)
        return self.weight * hidden_states




        


