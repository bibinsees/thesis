import torch
import torch.nn as nn


def SimCLR_loss(feats,temperature):
    cos_sim = nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    print(cos_sim)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    print('\n')

    print(self_mask)
    cos_sim.masked_fill_(self_mask, -9e15)
    print('\n')

    print(cos_sim)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    print('\n')
    print(pos_mask)
    print('\n')
    print(torch.logsumexp(cos_sim, dim=-1))
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll