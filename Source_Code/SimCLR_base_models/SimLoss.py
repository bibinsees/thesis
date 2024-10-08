import torch
import torch.nn as nn


def SimCLR_loss(feats, temperature):
    # Calculate cosine similarity
    cos_sim = nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Accuracy calculations
    comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                          cos_sim.masked_fill(pos_mask, -9e15)],
                         dim=-1)
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

    # Calculate accuracy metrics
    acc_top1 = (sim_argsort == 0).float().mean()
    acc_top5 = (sim_argsort < 5).float().mean()
    mean_pos = 1 + sim_argsort.float().mean()

    return nll, acc_top1.item(), acc_top5.item(), mean_pos.item()
