import os
import torch
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import matplotlib
matplotlib.use('Qt5Agg')

### The following code simply uses ImageDaraFolder to create the train and val datasets
# Then it creates

#####
data_path = "/home/nim/Downloads/cats_and_dogs/train"

###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


transform = transforms.Compose([
    transforms.Resize((input_size,input_size)), # (h,w)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset = ImageFolder(train_data_path, transform=transform)

val_dataset = ImageFolder(val_data_path, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6,
                                          shuffle=False, num_workers=2)

