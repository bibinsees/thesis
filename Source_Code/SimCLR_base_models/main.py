import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

from SimLoss import SimCLR_loss
from helper import plot_curves, accuracy, set_seed, split_dataset
from train import train
from model import Resnet
from dataloader import ImageDataset

set_seed(42)

image_dir = r"../../tiff_experiment_unsupervised_data/combined"

dataset = ImageDataset(image_dir=image_dir)
batch_size = 16



# Split the dataset with 20% for validation
val_percentage = 0.2
train_dataset, val_dataset = split_dataset(dataset, val_percentage)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=0) #num_workers=os.cpu count() using cluster gpu
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

for i, (image1, image2) in enumerate(train_loader):
    print(f"Batch {i}:")
    print(f"  Image1: {image1.shape}")
    print(f"  Image2: {image2.shape}")
    break

model = Resnet(96*96)

model,results = train(train_loader,model,2,device='cpu')
