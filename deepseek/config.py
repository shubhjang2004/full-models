import torch
from dataclasses import dataclass

@dataclass
class Modelargs:
    dim:int=1024
    vocab_size:int=50256
    seqlen:int=256
    dtype:str="bf16"
    n_heads:int=16
    q_lora_rank:int=0
    kv_lora_rank:int= 256
    qk_nope_head_dim:int=64
    qk_rope_head_dim:int=64
    v_head_dim:int=64
    num_experts:int=8
    top_k:int=2
    n_layers:int=16
    num_epochs:int=4
    


    


