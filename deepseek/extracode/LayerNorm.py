import torch
import torch.nn as nn
import torch.nn.functional as F

class Layernorm(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.dim=dim
        self.eps=1e-5
        self.alpha=nn.Parameter(torch.empty(self.dim))
        self.beta=nn.Parameter(torch.empty(self.dim))
        torch.nn.init.normal_(self.alpha,mean=0.0,std=0.02)
        torch.nn.init.zeros_(self.beta)

    def forward(self,x):
        x_mean=torch.mean(x,dim=-1,keepdim=True)   
        x_std=torch.std(x,dim=-1,keepdim=True)
        x_norm=self.alpha*(x-x_mean)/(x_std+self.eps)
        x_norm += self.beta
        return x_norm 