import torch
from torch import nn, optim


from SimLoss import SimCLR_loss
from helper import accuracy
import tqdm

def train(train_loader,model,epochs,lr=5e-4, temperature=0.07, weight_decay=1e-4,device='cuda',validate=False):
    if not validate:
        model = model.train().to(device)
    else:
        model = model.eval().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=lr / 50)

    
    train_losses = []
    train_top1_accs = []
    train_top5_accs = []
    for epoch in range(epochs):
        total_loss = 0
        total_top1_acc = 0.0
        total_top5_acc = 0.0
        for imgs1, imgs2 in train_loader: #for batch in tqdm(train_loader, desc="Training", leave=False):
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
            imgs = torch.cat((imgs1, imgs2), dim=0)
            print(imgs.shape)
            feats = model.forward(imgs)
            print(feats.shape)
            
            loss = SimCLR_loss(feats=feats,temperature=temperature)
            if not validate:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            total_loss += loss.item()
            
            target = torch.cat((torch.arange(imgs1.size(0)), torch.arange(imgs2.size(0)))).to(device)
            print(imgs1.shape,imgs.shape)
            print(feats.shape,target.shape)
            top1_acc, top5_acc = accuracy(feats, target, topk=(1, 5))
            total_top1_acc += top1_acc.item()
            total_top5_acc += top5_acc.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_top1_acc = total_top1_acc / len(train_loader)
        avg_top5_acc = total_top5_acc / len(train_loader)
        train_losses.append(avg_loss)
        train_top1_accs.append(avg_top1_acc)
        train_top5_accs.append(avg_top5_acc)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Top-1 Acc: {avg_top1_acc:.2f}% | "
              f"Top-5 Acc: {avg_top5_acc:.2f}% | ")

    return model,(train_losses,train_top1_accs,train_top5_accs)