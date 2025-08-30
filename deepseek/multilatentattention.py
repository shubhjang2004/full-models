import torch
import torch.nn as nn
import torch.nn.functional as F
from extracode.LayerNorm import Layernorm
from rope import Rope
from config import Modelargs
class MLA(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.dim=args.dim
        self.seqlen=args.seqlen
        self.n_heads=args.n_heads
        self.q_lora_rank=args.q_lora_rank
        self.kv_lora_rank=args.kv_lora_rank
        self.qk_nope_head_dim=args.qk_nope_head_dim
        self.qk_rope_head_dim=args.qk_rope_head_dim
        self.v_head_dim=args.v_head_dim
        self.qk_head_dim=args.qk_nope_head_dim+args.qk_rope_head_dim

        if args.q_lora_rank==0:
            self.Wq=nn.Linear(args.dim,args.n_heads*self.qk_head_dim)
        else:
            self.Wq_a=nn.Linear(args.dim,args.q_lora_rank)
            self.Q_norm=Layernorm(args.q_lora_rank)
            self.Wq_b=nn.Linear(self.q_lora_rank,self.n_heads*(self.qk_nope_head_dim+self.qk_rope_head_dim))
        self.W_kv_krope=nn.Linear(self.dim,self.kv_lora_rank+self.qk_rope_head_dim)
        self.W_kv_b=nn.Linear(self.kv_lora_rank,self.n_heads*(self.qk_nope_head_dim+self.v_head_dim))
        self.W_out=nn.Linear(self.dim,self.dim)
        
        self.softmax_scale=self.qk_head_dim**(-0.5)
        #mask
        self.register_buffer("bias",torch.tril(torch.ones(args.seqlen,args.seqlen)).view(1,1,args.seqlen,args.seqlen))
        

    def forward(self,x):
        B,T,C=x.size()
        posn_rope=Rope(self.qk_rope_head_dim,T)
        assert self.dim==C and C%self.n_heads==0
        if self.q_lora_rank==0:
            q_combined=self.Wq(x)
        else:
            q_lora=self.Wq_a(x)
            q_lora=self.Q_norm(q_lora)
            q_combined=self.Wq_b(q_lora)

        q_combined=q_combined.view(B,T,self.n_heads,(self.qk_rope_head_dim+self.qk_nope_head_dim)).transpose(1,2)
        q_rope,q_nope=torch.split(q_combined,(self.qk_rope_head_dim,self.qk_nope_head_dim),dim=-1)

        q_rope=posn_rope(q_rope)
        query=torch.cat([q_nope,q_rope],dim=-1)

        ## handles key and values
        kv_lora_k_rope=self.W_kv_krope(x)
        kv_lora,k_rope=torch.split(kv_lora_k_rope,(self.kv_lora_rank,self.qk_rope_head_dim),dim=-1)
        k_rope=k_rope.unsqueeze(2).expand(-1,-1,self.n_heads,-1).transpose(1,2)
        k_rope=posn_rope(k_rope)

        kv_combined=self.W_kv_b(kv_lora)
        kv_combined=kv_combined.view(B,T,self.n_heads,(self.qk_nope_head_dim+self.v_head_dim))
        k_nope,value=torch.split(kv_combined,(self.qk_nope_head_dim,self.v_head_dim),dim=-1)
        value=value.transpose(1,2)
        k_nope=k_nope.transpose(1,2)
        key=torch.cat([k_nope,k_rope],dim=-1)


        ## attn part 

        #attn =query@key.transpose(-2,-1)*self.softmax_scale
        #attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
        #attn=F.softmax(attn,dim=-1)
        # y=attn@value
        y=F.scaled_dot_product_attention(query,key,value,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.W_out(y)



          

   



                




