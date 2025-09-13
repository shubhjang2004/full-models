import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class BertArgs:
    vocab_size: int = 30522
    seqlen: int = 512
    dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    eps: float = 1e-12
    num_classes: int = 2



class Bert(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.bert=BertModel(args)
        self.dropout=nn.Dropout(args.dropout)
        self.classifier=nn.Linear(args.dim,args.num_classes)

    def forward(self,idx,attn_mask=None,token_type_ids=None,target=None):


        if attn_mask is not None:
            if token_type_ids is not None:
                pooled=self.bert(idx,attn_mask,token_type_ids)
            else:
                pooled=self.bert(idx,attn_mask)

        else:
            if token_type_ids is not None:
                pooled=self.bert(idx,token_type_ids)
            else:
                pooled=self.bert(idx)
          
        
        x=self.dropout(pooled)
        
        if target is not None:
            logits=self.classifier(x)
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        else:
            logits=self.classifier(x)
            loss=None

        return logits,loss        
        

        


class BertModel(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.embeddings=BertEmbeddings(args)
        self.encoder=BertEncoder(args)
        self.pooler=BertPooler(args)

    def forward(self,idx,attn_mask=None,token_type_ids=None):
        if token_type_ids is not None:
            x=self.embeddings(idx,token_type_ids)
        else:
            x=self.embeddings(idx)   
        
        if attn_mask is not None:
            x=self.encoder(x,attn_mask)

        else:
            x=self.encoder(x)

        x=self.pooler(x)

        return x        



class BertEmbeddings(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.word_embeddings=nn.Embedding(args.vocab_size,args.dim,padding_idx=0)
        self.position_embeddings=nn.Embedding(args.seqlen,args.dim)
        self.token_type_embeddings=nn.Embedding(2,args.dim)
        self.LayerNorm=nn.LayerNorm(args.dim,args.eps,elementwise_affine=True)
        self.dropout=nn.Dropout(args.dropout,inplace=False)


    def forward(self,idx,token_type_ids=None):
        B,T=idx.size()    
        embed_tokens=self.word_embeddings(idx)

        posn=torch.arange(0,T,dtype=torch.long,device=idx.device)
        posn = posn.unsqueeze(0).expand(B, T)  # (B, T)
        posn_tokens=self.position_embeddings(posn)
        

         # 3. Token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros((B, T), dtype=torch.long, device=idx.device)

        tokentype_tokens = self.token_type_embeddings(token_type_ids)  # (B, T, dim)

        x = embed_tokens + posn_tokens + tokentype_tokens    

        x=self.LayerNorm(x)
        x=self.dropout(x)
        return x


class BertEncoder(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.layer=nn.ModuleList([BertLayer(args) for _ in range(args.n_layers)])

    def forward(self,x,attn_mask=None):
        if attn_mask is not None:
            for layer in self.layer:
                x=layer(x,attn_mask)
        else:
            for layer in self.layer:
                x=layer(x)

        return x                 


class BertLayer(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()

        self.attention=BertAttention(args)
        self.intermediate=BertIntermediate(args)
        self.output=BertOutput(args)

    def forward(self,x,attn_mask=None):
        #---passing through attention +residual----
        if attn_mask is not None:
            x=self.attention(x,attn_mask) 
        else:
            x=self.attention(x)

        #---passing through feed forward----

        hidden=self.intermediate(x)
        output=self.output(x,hidden)
        return output


class BertAttention(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.self = BertSdpaSelfAttention(args)
        self.output=BertSelfOutput(args)
   
    def forward(self,x,attn_mask=None):
        if attn_mask is not None:
            hidden=self.self(x,attn_mask)
        else:
            hidden=self.self(x)

        y=self.output(x,hidden)

        return y        




class BertSdpaSelfAttention(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.query=nn.Linear(args.dim,args.dim)
        self.key=nn.Linear(args.dim,args.dim)
        self.value=nn.Linear(args.dim,args.dim)
        self.dropout=nn.Dropout(args.dropout)

        assert args.dim % args.n_heads==0, f"shape mismatch head nos"
        self.n_heads=args.n_heads
        self.head_dim=args.dim//args.n_heads
        self.softmax_scale=self.head_dim**-0.5

    def forward(self,x,attn_mask=None):

        B,T,C=x.size()
        assert C%self.n_heads==0 , f"shape mismatch"
        query=self.query(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        key=self.key(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)
        value=self.value(x).view(B,T,self.n_heads,C//self.n_heads).transpose(1,2)

        attn=query@key.transpose(-1,-2)*self.softmax_scale
        if attn_mask is not None:
            mask = attn_mask[:, None, None, :]  # [B, 1, 1, T]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn=F.softmax(attn,dim=-1)
        out=attn@value
        out=out.transpose(1,2).contiguous().view(B,T,C)
        out=self.dropout(out)
        return out


class BertSelfOutput(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.dense=nn.Linear(args.dim,args.dim)
        self.LayerNorm=nn.LayerNorm(args.dim,eps=args.eps,elementwise_affine=True)
        self.dropout=nn.Dropout(args.dropout,inplace=False)

    def forward(self,input,attn_output):
        attn_output=self.dense(attn_output)
        attn_output=self.dropout(output)

        output=self.LayerNorm(input+attn_output)
        
        return output  




class BertIntermediate(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.dense=nn.Linear(args.dim,4*args.dim)
        self.intermediate_act_fn=nn.GELU()

    def forward(self,input):
        hidden_state=self.dense(input)
        hidden_state=self.intermediate_act_fn(hidden_state)
        return hidden_state    


class BertOutput(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.dense=nn.Linear(4*args.dim,args.dim)
        self.LayerNorm=nn.LayerNorm(args.dim,eps=args.eps,elementwise_affine=True)
        self.dropout=nn.Dropout(args.dropout,inplace=False)

    def forward(self,input,hidden_state):
        hidden_state=self.dense(hidden_state)
        hidden_state=self.dropout(hidden_state)
        output=self.LayerNorm(input+hidden_state)
        
        return output


class BertPooler(nn.Module):
    def __init__(self,args:BertArgs):
        super().__init__()
        self.dense=nn.Linear(args.dim,args.dim)
        self.activation=nn.Tanh()


    def forward(self,x):
        cls_token = x[:, 0] 
        pooled=self.dense(cls_token)
        pooled=self.activation(pooled)
        return pooled
              

                
