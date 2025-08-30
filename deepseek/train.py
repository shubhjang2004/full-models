import torch
import os
import json
from data import train_loader,val_loader


def train_model(model,train_loader,val_loader,num_epochs,optimizer,device="cpu",
                save_path="CheckPoint/model.pt",log_path="Checkpoints/log.json",
                save_weights_only=False):
    
    model=model.to(device)
    train_losses=[]
    val_losses=[]
    os.makedies(os.path.dirname(save_path),exist_ok=True)
    for epoch in range(num_epochs):
        print(f"\nepoch{epoch}/{num_epochs}")

        model.train()
        train_epoch_loss=0.0
        for batch in train_loader:
            input=batch["input"]
            target=batch['target']
            input,target=input.to(device),target.to(device)
            optimizer.zero_grad()
            loss,logits=model(input,target)
            loss.backward()
            optimizer.step()

            train_epoch_loss +=loss.item
            
        avg_train_loss=train_epoch_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"train loss:{avg_train_loss:.4f}")
        

        model.eval()
        val_epoch_loss=0.0
        with torch.no_grad():
            for batch in val_loader:
                input=batch["input"]
                target=batch["target"]
            
                input,target=input.to(device),target.to(device)
                loss,logits=model(input,target)
                val_epoch_loss +=loss.item

            avg_val_loss= val_epoch_loss/len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"val_loss:{avg_val_loss:.4f}")

            checkpoint = model.state_dict() if save_weights_only else model
            torch.save(checkpoint, save_path)
            print((f"Model saved to: {save_path}"))

            with open(log_path, "w") as f:
                json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=4)        


#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

"""
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device="cpu",
    num_epochs=10,
    save_path="checkpoints/model_epoch.pt",  # file path
    log_path="checkpoints/loss_log.json",    # JSON log
    save_weights_only=True                   # change to False to save full model
)

"""


for batch in train_loader :
    print(batch)
    break


