import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
#from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import  AdamW
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

model_name="gpt2"
model=AutoModelForCausalLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)


# to see it cuda is available or not
device="cuda" if torch.cuda.is_available() else "cpu"
device

#  url for downloading the daatset !wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json4

dataset=load_dataset("json",data_files="alpaca_data.json")

# dataset is in dictionary to convert it text so model can process
def format_chat(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n"
    return {"text": prompt + example["output"]}


dataset=dataset["train"].map(format_chat)


# to pad we are using tokenizer.eos_token_id
dataset_raw=dataset["text"]
pad_token_id=tokenizer.eos_token_id


maxlen=1024
encoded_dataset=[]


# without padding and trucatioon
for example in dataset_raw:
    token_ids=tokenizer.encode(example,add_special_tokens=False) 
    encoded_dataset.append(token_ids)

# after padding and truncation
processed_dataset=[]
attention_masks=[]
for example in encoded_dataset:
    if (len(example)>maxlen):
       example=example[:maxlen]
    if(len(example)<maxlen):
       while(len(example)<maxlen):
          example.append(pad_token_id)
    attention_mask=[1 if t!=pad_token_id else 0 for t in example]
    processed_dataset.append(example)
    attention_masks.append(attention_mask)


labels=[]
for token in processed_dataset:
    shifted=token[1:]+[pad_token_id]
    shifted=[token_id if token_id != pad_token_id else -100 for token_id in shifted]
    labels.append(shifted)


class ChatDataset(Dataset):
    def __init__(self, input_ids, labels, attention_masks):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_masks, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx]
        }
     
   
train_dataset=ChatDataset(processed_dataset,labels,attention_masks)
dataset_len = len(train_dataset)
# Split sizes: 90% train, 10% val
train_size = int(0.95 * dataset_len)
val_size   = dataset_len - train_size
# Optional: for reproducibility
generator = torch.Generator().manual_seed(42)
# Perform random split
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


optimizer = AdamW(model.parameters(), lr=5e-5)
grad_clip=1.0
scaler = GradScaler()



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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)


                       # forward pass with autocast for FP16
            with autocast():
                output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                         )
                loss = output.loss

            # scale the loss and backward
            scaler.scale(loss).backward()

            # gradient clipping (optional)
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # optimizer step
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item()
            if(train_step%200==0):
                train_losses.append(loss.item())

                print(f"train_loss as step {train_step} : {loss.item()}")



        train_avg_loss=running_train_loss/len(train_loader)
        train_epoch_losses.append(train_avg_loss)
        print(f"train_loss for epoch {epoch+1}:{train_avg_loss}")
        model.eval()

        with torch.no_grad():
            for val_step,batch in enumerate(val_loader):


                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # forward pass with autocast for FP16

                with autocast():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                         )
                    loss = output.loss




                running_val_loss +=loss.item()


                if(val_step%40==0):
                    val_losses.append(loss.item())
                    print(f"val_loss as step {val_step} : {loss.item()}")
            val_avg_loss= running_val_loss/len(val_loader)
            val_epoch_losses.append(val_avg_loss)

            print(f"val loss for epoch {epoch+1} : {val_avg_loss}")


        save_dir = "/content/drive/MyDrive/bart_latest"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model and tokenizer saved at {save_dir}")

    return {
        "train_step_losses": train_losses,
        "train_epoch_losses": train_epoch_losses,
        "val_step_losses": val_losses,
        "val_epoch_losses": val_epoch_losses
    }
