import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from Classification.OCT_Classification.data_utils.dataset_transforms import get_transform
from Classification.OCT_Classification.data_utils.data_loader_utils import create_dataloader, get_class_dist_from_dataloader, get_class_dist_from_dataset
# it runs the whole module if there's no if == main


### The following code simply uses ImageDaraFolder to create the train and val datasets
# Then it creates dataloaders

base_split_folder = '/home/nim/Downloads/cats_and_dogs'
base_split_folder = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_0_035'
train_data_path = os.path.join(base_split_folder, 'train')
val_data_path = os.path.join(base_split_folder, 'val')

transform = get_transform(input_size=224)
train_dataset = ImageFolder(train_data_path, transform=transform)
val_dataset = ImageFolder(val_data_path, transform=transform)
train_loader, val_loader = create_dataloader(train_dataset, val_dataset, bs_train=48, bs_val=48)

######

# it = iter(train_loader)
# it.__next__()[0].shape # torch.Size([24, 3, 224, 224])

num_classes = 4
### ChatGPT approach. Problem - doesn't use all samples (indices) for training
class_indices, class_dist, idx2class = get_class_dist_from_dataset(train_dataset, num_classes)
class_counts = get_class_dist_from_dataloader(train_loader, num_classes)

min_class_size = min(len(class_indices[0]), len(class_indices[1]))

# Use SubsetRandomSampler to create balanced samplers for each class
balanced_indices = []
for class_idx in class_indices.values():
    balanced_indices.extend(class_idx[:min_class_size])
# balanced_indices = [idx for class_idx in class_indices.values() for idx in class_idx[:min_class_size]]

# Create a DataLoader with the balanced sampler
train_loader = DataLoader(train_dataset, batch_size=6, sampler=SubsetRandomSampler(balanced_indices), num_workers=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=2)

# Calculates the Class-dsitribution

