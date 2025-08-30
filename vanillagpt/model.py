import torch
import torch.nn as nn
import torch.nn.functional as F
from block import Config
from block import GPTNeoBlock



args=Config()
class Transformer(nn.Module):
    def __init__(self,args:Config):
        super().__init__()
        self.transformer=nn.ModuleDict({
            "wte":nn.Embedding(args.vocab_size,args.dim),
            "wpe":nn.Embedding(args.seqlen,args.dim),
            "drop":nn.Dropout(args.dropout),
            "h":nn.ModuleList([GPTNeoBlock(args) for _ in range(args.n_layers)]),
            "ln_f":nn.LayerNorm(args.dim)
        })
    
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        #self.lm_head.weight = self.transformer.wte.weight  # tie weights after creation
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.weight_initalize)

    def forward(self,idx,target=None):
        B,T=idx.size()
        wte=self.transformer.wte(idx)
        posn=torch.arange(0,T,dtype=torch.long,device=idx.device).unsqueeze(0)
        wpe=self.transformer.wpe(posn)
        x=self.transformer.drop(wte+wpe)


        ## posn change expansion in batch dimension 

        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)

        if target is not None:
            loss=F.cross_entropy(logits.view(-1,x.size(-1)),target.view(-1),ignore_index=-1,reduction="mean")

            return logits,loss
        return logits
    
    def weight_initalize(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def get_params(self):
        return sum(p.numel() for p in self.parameters())    


    """
    def generate(self,input,maxnew_tokens:int,device="cpu"):
        self.eval()
        input= input.to(device)
        full_output=input.clone()
        for _ in range(maxnew_tokens):
            if(input.shape[1]>args.seqlen):
                input=input[:,-args.seqlen:]

            logits=self(input)
            last_token_logits=logits[:,-1,:]
            probs=F.softmax(last_token_logits,dim=-1)
            next_token=torch.multinomial(probs,num_samples=1)
            input=torch.cat([input,next_token],dim=1)
            full_output=torch.cat([full_output,next_token],dim=1)



        return full_output
    """
    """
    
    def generate(self, idx, max_new_tokens):
        output=idx.clone()
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            
            idx_cond = idx[:, -args.seqlen:]
            # get the predictions
            logits= self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            output=torch.cat((output,idx_next),dim=-1)
        return output
        """
    def generate(self, input_ids, max_new_tokens:int, device="cpu"):
        self.eval()
        self.to(device)
        input_ids = input_ids.to(device)
        full_output = input_ids.clone()

        max_seq_len = args.seqlen
        for _ in range(max_new_tokens):
           # truncate input to max seq len
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]

            logits = self(input_ids)  # [B, T, vocab_size]
            last_token_logits = logits[:, -1, :]  # [B, vocab_size]

            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B,1]

            input_ids = torch.cat([input_ids, next_token], dim=1)
            full_output = torch.cat([full_output, next_token], dim=1)

        return full_output








