import os
import torch
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import matplotlib
matplotlib.use('Qt5Agg')

### The following code simply uses ImageDaraFolder to create the train and val datasets
# Then it creates

#####
train_data_path = "/home/nim/Downloads/cats_and_dogs/train"
val_data_path = "/home/nim/Downloads/cats_and_dogs/val"

train_data_path = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_0_035/train'
val_data_path = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_0_035/val'

###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


transform = transforms.Compose([
    transforms.Resize((input_size,input_size)), # (h,w)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(train_data_path, transform=transform)
val_dataset = ImageFolder(val_data_path, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6,
                                          shuffle=False, num_workers=2)


class_indices = {0: [], 1: []}
for i, (data, label) in enumerate(train_dataset):
    class_indices[label].append(i)

min_class_size = min(len(class_indices[0]), len(class_indices[1]))

# Use SubsetRandomSampler to create balanced samplers for each class
balanced_indices = []
for class_idx in class_indices.values():
    balanced_indices.extend(class_idx[:min_class_size])


# Create a DataLoader with the balanced sampler
train_loader = DataLoader(train_dataset, batch_size=6, sampler=SubsetRandomSampler(balanced_indices), num_workers=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=2)