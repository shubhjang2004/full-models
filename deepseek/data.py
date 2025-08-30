import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader     
import tiktoken


with open(r"C:\Users\sjang\OneDrive\Desktop\deepseek\input.txt","r",encoding="utf-8") as f:
    train_text=f.read()


tokenizer=tiktoken.get_encoding("gpt2")
tokens=tokenizer.encode(train_text)
len_train=int(0.9*len(tokens))
train_tokens=tokens[:len_train]
val_tokens=tokens[:len_train]
        




class NonOverlappingDataset(Dataset):
    def __init__(self,tokens,block_size):
       
        total_len=(len(tokens)//block_size)*block_size
        self.tokens=tokens[:total_len]
        self.block_size=block_size
        self.num_chunks=len(self.tokens)//block_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self,idx):
        start=idx*self.block_size
        end=start+self.block_size
        chunk=self.tokens[start:end]

        return{
              #torch.tensor(chunk[:-1]),
              #torch.tensor(chunk[1:])
             "input": chunk[:-1],
             "target": chunk[1:]

        }  
    

## IF YOU WANT TO TRAIN THE MODEL ON LESS DATA SO YOU SAMPLE SAME TOKEN MULTIPLE TIMES JUST IN CONTEXT OF DIFFERENT TOKENS    
"""
class TextChunkDataset(Dataset):
   def __init__(self,tokens,block_size):
       self.tokens=tokens
       self.block_size=block_size
   def __len__(self):
       return len(self.tokens)-self.block_size-1
   def __getitem__(self,idx):
       chunk=self.tokens[idx:idx+self.block_size+1]
       return{
           "input":chunk[:-1],
           "target":chunk[1:]

       }      
"""


context_size=128
train_dataset=NonOverlappingDataset(train_tokens,block_size=context_size)
train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)

val_dataset=NonOverlappingDataset(val_tokens,block_size=context_size)
val_loader=DataLoader(val_dataset,batch_size=4,shuffle=True)





