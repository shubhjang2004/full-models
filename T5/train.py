import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from datasets import load_dataset,DatasetDict
from torch.utils.data import Dataset,DataLoader
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType


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


def collate_fn(batch):
    input_ids=batch["input_ids"]
    attention_mask=batch["attention_mask"]
    labels=batch["labels"]

    new_batch = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        return_tensors="pt"
    )
    max_label_len = max(len(l) for l in labels)
    padded_labels = [l + [pad_id] * (max_label_len - len(l)) for l in labels]
    labels = torch.tensor(padded_labels, dtype=torch.long)
    labels[labels == tokenizer.pad_token_id] = -100

    new_batch["labels"] = labels

    return new_batch




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
peft_model = get_peft_model(model_soft, peft_config)

