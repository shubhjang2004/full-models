import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from datasets import load_dataset,DatasetDict
from torch.utils.data import Dataset,DataLoader
from peft import get_peft_model, PromptTuningConfig, LoraConfig,PromptTuningInit, TaskType
from torch.optim import AdamW

device="cuda" if torch.cuda.is_available() else "cpu"

model_name="t5-small"
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

dataset=load_dataset("imdb")

new_dataset=DatasetDict({
    "train":dataset["train"],
    "val":dataset["test"].shuffle(seed=42).select(range(5000))
    
})

idx_to_label={0:"negative",1:"positive"}





max_input_len=512
max_target_len=2
pad_id = tokenizer.pad_token_id

def proprocess(dataset):
    input_text=["classify sentiment: "+t for t in dataset["text"]]
    inputs=tokenizer(input_text,max_length=max_input_len,truncation=True, padding="max_length")
    label=[idx_to_label[label] for label in dataset["label"]]
    with tokenizer.as_target_tokenizer():
        label=tokenizer(label,max_length=max_target_len,truncation=True, padding="max_length")
      
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": label["input_ids"],
    }


token_dataset=new_dataset.map(proprocess,batched=True,remove_columns=new_dataset["train"].column_names)

        

tokens_train=DatasetDict({
    "input_ids":token_dataset["train"]["input_ids"],
    "attention_mask":token_dataset["train"]["attention_mask"],
    "labels":token_dataset["train"]["labels"]
})

tokens_val=DatasetDict({
    "input_ids":token_dataset["val"]["input_ids"],
    "attention_mask":token_dataset["val"]["attention_mask"],
    "labels":token_dataset["val"]["labels"]
})


class ImdbDataset(Dataset):
    def __init__(self,token_dataset):
        super().__init__()
        self.input_ids=torch.tensor(token_dataset["input_ids"],dtype=torch.long)
        self.attention_mask=torch.tensor(token_dataset["attention_mask"],dtype=torch.long)
        self.labels=torch.tensor(token_dataset["labels"],dtype=torch.long)
        self.labels[self.labels==pad_id]=-100
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return{
            "input_ids":self.input_ids[idx],
            "attention_mask":self.attention_mask[idx],
            "label":self.labels[idx]
        }    





train_dataset=ImdbDataset(tokens_train)
val_dataset=ImdbDataset(tokens_val)

train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=8,shuffle=False)




peft_config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type=TaskType.SEQ_2_SEQ_LM,   # T5 is encoder-decoder / seq2seq
    prompt_tuning_init=PromptTuningInit.TEXT,  # or RANDOM
    prompt_tuning_init_text="classify sentiment:", # used if TEXT init
    num_virtual_tokens=30,
    token_dim=model.config.d_model,   # embedding size
    tokenizer_name_or_path=model_name,
)

model_soft=AutoModelForSeq2SeqLM.from_pretrained(model_name)
model_soft = get_peft_model(model_soft, peft_config)


# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # works for T5, BART, etc.
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model_lora=AutoModelForSeq2SeqLM.from_pretrained(model_name)

model_lora = get_peft_model(model_lora, lora_config)



optimizer_soft = AdamW(model_soft.parameters(), lr=5e-5, weight_decay=0.01)
optimizer_lora = AdamW(model_lora.parameters(), lr=5e-5, weight_decay=0.01)

def model_train(model,device,train_loader,val_loader,num_epochs,optimizer,grad_clip=None):
    model=model.to(device)
    train_losses=[]
    val_losses=[]
    train_epoch_losses=[]
    val_epoch_losses=[]


    for epoch in range(num_epochs):
        print(f"epoch: {epoch+1}/{num_epochs}")
        model.train()
        running_val_loss=0
        running_train_loss=0
        for train_step, batch in enumerate(train_loader):
            
           

            optimizer.zero_grad()
            input_ids=batch["input_ids"]
            attn_mask=batch["attention_mask"]
            labels=batch["label"]
            output=model(input_ids=input_ids.to(device),
                        attention_mask=attn_mask.to(device),
                        labels=labels.to(device))
            
            loss=output.loss
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_train_loss += loss.item()
            if(train_step%100==0):
                train_losses.append(loss.item())
                if(train_step%130==0):
                    print(f"train_loss as step {train_step} : {loss.item()}")



        train_avg_loss=running_train_loss/len(train_loader)
        train_epoch_losses.append(train_avg_loss)
        print(f"train_loss for epoch {epoch}:{train_avg_loss}")
        model.eval()

        with torch.no_grad():
            for val_step,batch in enumerate(val_loader):
                          
                input_ids=batch["input_ids"]
                attn_mask=batch["attention_mask"]
                labels=batch["label"]
                output=model(input_ids=input_ids.to(device),
                            attention_mask=attn_mask.to(device),
                            labels=labels.to(device))
                loss=output.loss
                

                running_val_loss +=loss.item()
                val_losses.append(loss.item())

                if(val_step%20==0):
                    print(f"val_loss as step {val_step} : {loss.item()}")
            val_avg_loss= running_val_loss/len(val_loader)
            val_epoch_losses.append(val_avg_loss)

            print(f"val loss for epoch {epoch} : {val_avg_loss}")

    return {
        "train_step_losses": train_losses,
        "train_epoch_losses": train_epoch_losses,
        "val_step_losses": val_losses,
        "val_epoch_losses": val_epoch_losses
    }



soft_prompt_results=model_train(model_soft,device=device,train_loader=train_loader,val_loader=val_loader,num_epochs=4,optimizer=optimizer_soft,grad_clip=1.0)

lora_results=model_train(model_lora,device=device,train_loader=train_loader,val_loader=val_loader,num_epochs=4,optimizer=optimizer_lora,grad_clip=1.0)