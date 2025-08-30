import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Modelargs
from Block import MLPBlock,MOEBlock
from extracode.LayerNorm import Layernorm
import tiktoken

class Transformer(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.args=args
        assert args.vocab_size is not None
        assert args.seqlen is not None


        self.transformer= nn.ModuleDict({
            "embedding":nn.Embedding(args.vocab_size,args.dim),
            "blocks": nn.ModuleList([
                     MOEBlock(args) if i in {2, 4, 6, 8} else MLPBlock(args)
                            for i in range(args.n_layers)
                               ]),
            "ln_fn":nn.LayerNorm(args.dim)
        })

        self.lm_head=nn.Linear(args.dim,args.vocab_size)

        self.lm_head.weight = self.transformer["embedding"].weight 
        self.apply(self.weight_initialize)



    def forward(self,idx,target=None):
        B,T=idx.size()
        x=self.transformer.embedding(idx)
        for block in self.transformer.blocks:
            x=block(x)
        x=self.transformer.ln_fn(x)
        logits=self.lm_head(x)
        
        
        if target is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1),ignore_index=1,reduction='mean')
            return logits,loss
           
        return logits
    
    ## to find no of parameters 
    def get_params(self):
        n_params=sum(p.numel() for p in self.parameters())    
        return n_params
    
    # to initialise the weights
    def weight_initialize(self,Module):
        if isinstance(Module,nn.Linear):
            torch.nn.init.normal_(Module.weight,mean=0.0,std=0.02)
            if Module.bias is not None:
                torch.nn.init.zeros_(Module.bias)
        elif isinstance(Module,nn.Embedding):
            torch.nn.init.normal_(Module.weight,mean=0.0,std=0.02)


    ## to genearte new tokens bt the model        
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.args.seqlen else idx[:, -self.args.seqlen:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


        
