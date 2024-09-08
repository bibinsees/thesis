import torch
from torch import nn, optim


from SimLoss import SimCLR_loss
import tqdm

def train(train_loader, model, epochs, lr=5e-4, temperature=0.07, weight_decay=1e-4, device='cuda', validate=False):
    if not validate:
        model = model.train().to(device)
    else:
        model = model.eval().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=lr / 50)

    train_losses = []
    train_top1_accs = []
    train_top5_accs = []
    mean_positions = []

    for epoch in range(epochs):
        total_loss = 0
        total_top1_acc = 0.0
        total_top5_acc = 0.0
        total_mean_pos = 0.0

        for imgs1, imgs2 in train_loader:
            imgs1, imgs2 = imgs1.to(device), imgs2.to(device)
            imgs = torch.cat((imgs1, imgs2), dim=0)
            feats = model(imgs)

            # Compute the loss and accuracy
            loss, acc_top1, acc_top5, mean_pos = SimCLR_loss(feats, temperature)

            if not validate:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            total_loss += loss.item()
            total_top1_acc += acc_top1
            total_top5_acc += acc_top5
            total_mean_pos += mean_pos

        avg_loss = total_loss / len(train_loader)
        avg_top1_acc = total_top1_acc / len(train_loader)
        avg_top5_acc = total_top5_acc / len(train_loader)
        avg_mean_pos = total_mean_pos / len(train_loader)

        train_losses.append(avg_loss)
        train_top1_accs.append(avg_top1_acc)
        train_top5_accs.append(avg_top5_acc)
        mean_positions.append(avg_mean_pos)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Top-1 Acc: {avg_top1_acc:.2f}% | "
              f"Top-5 Acc: {avg_top5_acc:.2f}% | "
              f"Mean Position: {avg_mean_pos:.2f}")

    return model, (train_losses, train_top1_accs, train_top5_accs, mean_positions)