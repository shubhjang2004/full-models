import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification,BertTokenizer
from datasets import load_dataset,DatasetDict
from torch.utils.data import Dataset,DataLoader
from peft import get_peft_model, PromptTuningConfig, LoraConfig,PromptTuningInit, TaskType
from torch.optim import AdamW

device="cuda" if torch.cuda.is_available() else "cpu"


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load dataset
dataset = load_dataset("imdb")
new_dataset = DatasetDict({
    "train": dataset["train"],
    "val": dataset["test"].shuffle(seed=42).select(range(5000))
})

num_virtual_tokens = 30
max_input_len_soft = 512 - num_virtual_tokens
max_input_len_lora=512

# Preprocessing for soft function
def preprocess_bert_soft(dataset):
    input_text = ["classify sentiment: " + t for t in dataset["text"]]
    inputs = tokenizer(input_text, max_length=max_input_len_soft, truncation=True, padding="max_length")
    labels = dataset["label"]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }
# Preprocessing for lora  function
def preprocess_bert_lora(dataset):
    input_text = ["classify sentiment: " + t for t in dataset["text"]]
    inputs = tokenizer(input_text, max_length=max_input_len_lora, truncation=True, padding="max_length")
    labels = dataset["label"]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

token_dataset_soft = new_dataset.map(preprocess_bert_soft, batched=True, remove_columns=new_dataset["train"].column_names)
token_dataset_lora = new_dataset.map(preprocess_bert_lora, batched=True, remove_columns=new_dataset["train"].column_names)


# Create PyTorch dataset
class ImdbDataset(Dataset):
    def __init__(self, token_dataset):
        self.input_ids = torch.tensor(token_dataset["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(token_dataset["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(token_dataset["labels"], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Datasets & dataloaders FOR SOFT
tokens_train_soft = token_dataset_soft["train"]
tokens_val_soft = token_dataset_soft["val"]

train_dataset_soft = ImdbDataset(tokens_train_soft)
val_dataset_soft = ImdbDataset(tokens_val_soft)

train_loader_soft = DataLoader(train_dataset_soft, batch_size=16, shuffle=True)
val_loader_soft = DataLoader(val_dataset_soft, batch_size=32, shuffle=False)

# Datasets & dataloaders for LORA
tokens_train_lora = token_dataset_lora["train"]
tokens_val_lora = token_dataset_lora["val"]

train_dataset_lora = ImdbDataset(tokens_train_lora)
val_dataset_lora = ImdbDataset(tokens_val_lora)

train_loader_lora = DataLoader(train_dataset_lora, batch_size=16, shuffle=True)
val_loader_lora = DataLoader(val_dataset_lora, batch_size=32, shuffle=False)





peft_config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type="SEQ_CLS",  # for classification
    num_virtual_tokens=30,
    token_dim=model.config.hidden_size,  # use hidden_size for BERT
)

model_soft=BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_soft = get_peft_model(model_soft, peft_config)


lora_config = LoraConfig(
    r=8,                   # rank of LoRA matrices
    lora_alpha=32,         # scaling factor
    target_modules=["query", "value"],  # BERT attention modules
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"    # classification
)
model_lora=BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

model_lora = get_peft_model(model_lora, lora_config)



optimizer_soft = AdamW(model_soft.parameters(), lr=3e-6, weight_decay=0)
grad_clip = 1.0

optimizer_lora = AdamW(model_lora.parameters(), lr=5e-5, weight_decay=0)
grad_clip = 1.0


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
            labels=batch["labels"]
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

                print(f"train_loss as step {train_step} : {loss.item()}")



        train_avg_loss=running_train_loss/len(train_loader)
        train_epoch_losses.append(train_avg_loss)
        print(f"train_loss for epoch {epoch}:{train_avg_loss}")
        model.eval()

        with torch.no_grad():
            for val_step,batch in enumerate(val_loader):

                input_ids=batch["input_ids"]
                attn_mask=batch["attention_mask"]
                labels=batch["labels"]
                output=model(input_ids=input_ids.to(device),
                            attention_mask=attn_mask.to(device),
                            labels=labels.to(device))
                loss=output.loss


                running_val_loss +=loss.item()


                if(val_step%20==0):
                    val_losses.append(loss.item())
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


soft_prompt_results=model_train(model_soft,device=device,train_loader=train_loader_soft,val_loader=val_loader_soft,num_epochs=4,optimizer=optimizer_soft,grad_clip=1.0)

lora_results=model_train(model_lora,device=device,train_loader=train_loader_lora,val_loader=val_loader_lora,num_epochs=4,optimizer=optimizer_lora,grad_clip=1.0)