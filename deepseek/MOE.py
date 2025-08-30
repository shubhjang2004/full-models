import torch
import torch.nn as nn
import torch.nn.functional as F




class MLP(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.dim=dim
        self.W_up=nn.Linear(dim,4*dim)
        self.gelu=nn.GELU()
        self.W_down=nn.Linear(4*dim,dim)

    def forward(self,x):
        return (self.W_down(self.gelu(self.W_up(x))))



class Expert(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.dim=dim
        self.W_up=nn.Linear(dim,2*dim)
        self.gelu=nn.GELU()
        self.W_down=nn.Linear(2*dim,dim)

    def forward(self,x):
        return self.W_down(self.gelu(self.W_up(x)))


class SparseMOE(nn.Module):
    def __init__(self,dim:int,num_experts:int,top_k:int):
        super().__init__()

        self.router=NoisyRouter(dim,num_experts,top_k)
        self.experts=nn.ModuleList([(Expert(dim)) for _ in range(num_experts)])
        self.top_k=top_k
    def forward(self,x):
        gating_output,indices=self.router(x)
        final_output=torch.zeros_like(x)

        flat_x=x.view(-1,x.size(-1))
        flat_gating_output=gating_output.view(-1,gating_output.size(-1)) 

        for i,expert in enumerate(self.experts):
            expert_mask=(indices==i).any(dim=-1)
            flat_mask=expert_mask.view(-1)

            if flat_mask.any():
                x_input=flat_x[flat_mask]
                expert_output=expert(x_input)

                gating_scores=flat_gating_output[flat_mask,i].unsqueeze(1)
                weighted_output =gating_scores*expert_output

                flat_final_output = final_output.view(-1, final_output.size(-1))
                flat_final_output[flat_mask] += weighted_output
                final_output = flat_final_output.view_as(x)



        return final_output        


        

class Gaterouter(nn.Module):
    def __init__(self,dim:int,num_experts:int,top_k:int):
        super().__init__()
        self.top_k=top_k
        self.gate_layer=nn.Linear(dim,num_experts)
       
        

    def forward(self,x):
        logits=self.gate_layer(x)
        top_k_logits,top_k_indices=logits.topk(self.top_k,dim=-1)

        zeros=torch.full_like(logits,float("-inf"))
        zeros=zeros.scatter(-1,top_k_indices,top_k_logits)

        sparse_logits=F.softmax(zeros,dim=-1)
        return sparse_logits,top_k_indices


class NoisyRouter(nn.Module):
    def __init__(self,dim:int,num_experts:int,top_k:int):
        super().__init__()
        self.top_k=top_k
        self.gatelayer=nn.Linear(dim,num_experts)
        self.noisylayer=nn.Linear(dim,num_experts)

    def forward(self,x):
        logits=self.gatelayer(x)
        noise_logits=self.noisylayer(x)
        noise=torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits=logits+noise

        top_k_logits,top_k_indices=noisy_logits.topk(self.top_k,dim=-1)
        zeros=torch.full_like(logits,float("-inf"))
        zeros=zeros.scatter(-1,top_k_indices,top_k_logits)
        sparse_logits=F.softmax(zeros,dim=-1)
        return sparse_logits,top_k_indices 



     





