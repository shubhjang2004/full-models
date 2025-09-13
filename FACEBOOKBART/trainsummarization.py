from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader,random_split
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

device="cuda" if torch.cuda.is_available() else "cpu"

from transformers import BartForConditionalGeneration,BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer=BartTokenizer.from_pretrained("facebook/bart-base")

from datasets import load_dataset
ds = load_dataset("abisee/cnn_dailymail", "1.0.0")




max_src_len = 1024
max_tgt_len = 128
num_examples = 50000  # limit

# Prepare all examples using tokenizer's built-in padding/truncation
encoder_inputs = []
encoder_attn_mask = []
decoder_inputs = []
decoder_attn_mask = []
decoder_target = []

for i, example in enumerate(ds["train"]):
    if i >= num_examples:
        break

    # Encode source (article)
    enc = tokenizer(example["article"], max_length=max_src_len, padding='max_length', truncation=True)
    # Encode target (highlights)
    tgt = tokenizer(example["highlights"], max_length=max_tgt_len, padding='max_length', truncation=True)

    # encoder inputs
    encoder_inputs.append(enc['input_ids'])
    encoder_attn_mask.append(enc['attention_mask'])

    # decoder inputs: add bos token at start
    dec_input_ids = [tokenizer.bos_token_id] + tgt['input_ids'][:-1]  # shift right
    decoder_inputs.append(dec_input_ids)
    decoder_attn_mask.append([1 if t != tokenizer.pad_token_id else 0 for t in dec_input_ids])

    # labels: replace pad tokens with -100
    labels = tgt['input_ids']
    labels = [t if t != tokenizer.pad_token_id else -100 for t in labels]
    decoder_target.append(labels)


# PyTorch Dataset
class SummarizationChunk(Dataset):
    def __init__(self, enc_inputs, enc_attn_mask, dec_inputs, dec_attn_mask, target):
        self.enc_inputs = torch.tensor(enc_inputs, dtype=torch.long)
        self.enc_attn_mask = torch.tensor(enc_attn_mask, dtype=torch.long)
        self.dec_inputs = torch.tensor(dec_inputs, dtype=torch.long)
        self.dec_attn_mask = torch.tensor(dec_attn_mask, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return {
            "enc_inputs": self.enc_inputs[idx],
            "enc_attn_mask": self.enc_attn_mask[idx],
            "dec_inputs": self.dec_inputs[idx],
            "dec_attn_mask": self.dec_attn_mask[idx],
            "target": self.target[idx]
        }



train_dataset = SummarizationChunk(
    encoder_inputs, encoder_attn_mask, decoder_inputs, decoder_attn_mask, decoder_target
)

dataset_len = len(train_dataset)

# Split sizes: 90% train, 10% val
train_size = int(0.9 * dataset_len)
val_size   = dataset_len - train_size
# Optional: for reproducibility

generator = torch.Generator().manual_seed(42)
# Perform random split

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


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
            input_ids=batch["enc_inputs"].to(device)
            attention_mask=batch["enc_attn_mask"].to(device)
            decoder_input_ids=batch["dec_inputs"].to(device)
            decoder_attention_mask=batch["dec_attn_mask"].to(device)
            labels=batch['target'].to(device)

                       # forward pass with autocast for FP16
            with autocast():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
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
            if(train_step%100==0):
                train_losses.append(loss.item())

                print(f"train_loss as step {train_step} : {loss.item()}")



        train_avg_loss=running_train_loss/len(train_loader)
        train_epoch_losses.append(train_avg_loss)
        print(f"train_loss for epoch {epoch+1}:{train_avg_loss}")
        model.eval()

        with torch.no_grad():
            for val_step,batch in enumerate(val_loader):

                
                input_ids=batch["enc_inputs"].to(device)
                attention_mask=batch["enc_attn_mask"].to(device)
                decoder_input_ids=batch["dec_inputs"].to(device)
                decoder_attention_mask=batch["dec_attn_mask"].to(device)
                labels=batch['target'].to(device)
                # forward pass with autocast for FP16

                with autocast():
                    output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels
                   )
                    loss = output.loss

          
            

                running_val_loss +=loss.item()


                if(val_step%20==0):
                    val_losses.append(loss.item())
                    print(f"val_loss as step {val_step} : {loss.item()}")
            val_avg_loss= running_val_loss/len(val_loader)
            val_epoch_losses.append(val_avg_loss)

            print(f"val loss for epoch {epoch+1} : {val_avg_loss}")

        save_dir = "./bart_latest"   # same folder every time
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model and tokenizer saved at {save_dir}")    

    return {
        "train_step_losses": train_losses,
        "train_epoch_losses": train_epoch_losses,
        "val_step_losses": val_losses,
        "val_epoch_losses": val_epoch_losses
    }


bart_history= model_train(model=model,device=device,train_loader=train_loader,val_loader=val_loader,num_epochs=3,optimizer=optimizer,grad_clip=None)
    