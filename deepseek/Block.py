import torch
import torch.nn.functional as F
import torch.nn as nn
from extracode.LayerNorm import Layernorm
from MOE import SparseMOE,MLP
from multilatentattention import MLA




class MLPBlock(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ln_1=nn.LayerNorm(args.dim)
        self.MLA=MLA(args)
        self.ln_2=nn.LayerNorm(args.dim)
        self.MLP=MLP(args.dim)

    def forward(self,x):
        x=x+self.MLA(self.ln_1(x))
        x=x+self.MLP(self.ln_2(x))
        return x



class MOEBlock(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ln_1=nn.LayerNorm(args.dim)
        self.MLA=MLA(args)
        self.ln_2=nn.LayerNorm(args.dim)
        self.sparseMoe=SparseMOE(args.dim,args.num_experts,args.top_k)


    def forward(self,x):
        x=x+self.MLA(self.ln_1(x))
        x=x+self.sparseMoe(self.ln_2(x))

        return x     

