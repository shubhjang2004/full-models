import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class Rope(nn.Module):
    def __init__(self,dim:int,seq_len:int):
        super().__init__()
        self.seq_len=seq_len
        self.dim=dim
        self.base=10000

        self.freqs_cis=self.precompute_freqs()


    def forward(self,x):
        B,n_heads,T,head_dim=x.size()
        
        head_dim=x.size(-1)
        assert head_dim%2==0 
        x=x.reshape(B,n_heads,T,head_dim//2,2)
        x_complex=torch.view_as_complex(x)
        freqs=self.freqs_cis[:T].to(x.device) ## for inference time when seq is smaller
        freqs=freqs.unsqueeze(0).unsqueeze(0)
        x_out=x_complex*freqs
        
        x_out=torch.view_as_real(x_out)
        x_out=x_out.reshape(B,n_heads,T,head_dim)
        return x_out

        

    def precompute_freqs(self):
        half_dim=self.dim//2
        freqs=torch.arange(half_dim,dtype=torch.float32)
        freqs=1.0/(self.base**(freqs/half_dim))
        t=torch.arange(0,self.seq_len,dtype=torch.float32)
        freqs=torch.outer(t,freqs)
        return torch.polar(torch.ones_like(freqs),freqs)
    
     



             


