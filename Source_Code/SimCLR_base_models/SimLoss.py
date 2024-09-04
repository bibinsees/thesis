import torch
import torch.nn as nn


def SimCLR_loss(feats,temperature):
    cos_sim = nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

    '''if concanated image have dfifferent format use below one'''
    # Create target tensor
    #target = torch.arange(cos_sim.shape[0], device=cos_sim.device)
    #target[0::2] += 1
    #target[1::2] -= 1
    #index = target.reshape(cos_sim.shape[0], 1).long()

    # Prepare ground_truth_labels
    #ground_truth_labels = torch.zeros(cos_sim.shape[0], cos_sim.shape[0], device=cos_sim.device).long()
    #src = torch.ones(cos_sim.shape[0], cos_sim.shape[0], device=cos_sim.device).long()
    #ground_truth_labels = torch.scatter(ground_truth_labels, 1, index, src)
    #pos_mask = ground_truth_labels.bool()
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll