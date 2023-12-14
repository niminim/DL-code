import os
import torch
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import cv2

import matplotlib
matplotlib.use('Qt5Agg')

### The following code simply uses ImageDaraFolder to create the train and val datasets
# Then it creates dataloaders in different Class-Distributions

# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

#####
data_path = "/home/nim/Downloads/cats_and_dogs/train"

###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


transform = transforms.Compose([
    transforms.Resize((224,224)), # (h,w)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)), # (h,w)
    transforms.ToTensor(),
])

dataset = ImageFolder(data_path, transform=transform)
class_to_idx = dataset.class_to_idx
idx2class = {v: k for k, v in class_to_idx.items()}


def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
    return count_dict
count_dict = get_class_distribution(dataset)
print("Distribution of classes: \n", count_dict)

# plot the dataset distribution
plt.figure(figsize=(15,8))
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(dataset)]).melt(),
            x = "variable", y="value", hue="variable").set_title('Class Distribution')
####


### Create dataloaders using random_split (for get_class_distribution_loaders implementation must equal 1)
# Beware of having one transform for both datasets!
# Keep in mind the train-set is addressed as the whole dataset here.
# random_split transform the "dataset" into "dataset.Subset" which has no ".transform" --> random_split is not very useful

n_imgs_train = int(len(dataset.samples) * 0.8)
n_imgs_val = int(len(dataset.samples) - n_imgs_train)
train_dataset, val_dataset = random_split(dataset, (n_imgs_train, n_imgs_val))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                          shuffle=False, num_workers=2)
#

# Show Class-Distribution in the new random train-val splits
def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    for _, j in dataloader_obj:
        y_idx = j.item()
        y_lbl = idx2class[y_idx]
        count_dict[str(y_lbl)] += 1

    return count_dict

def plot_class_dir_loaders(train_dist, val_dist):
    # plots the class distribution in the train and val sets
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,7))
    sns.barplot(data = pd.DataFrame.from_dict([train_dist]).melt(),
                x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Train Set')
    sns.barplot(data = pd.DataFrame.from_dict([val_dist]).melt(),
                x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Val Set')

train_dist = get_class_distribution_loaders(train_loader, dataset)
val_dist = get_class_distribution_loaders(val_loader, dataset)
plot_class_dir_loaders(train_dist, val_dist)
#######


#########  SubsetRandomSampler
dataset_size = len(dataset)
dataset_indices = list(range(dataset_size))
np.random.shuffle(dataset_indices)
val_split_index = int(np.floor(0.2 * dataset_size))
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=train_sampler)
val_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=val_sampler)
val_loader.dataset.transform = transform_val # if necessary

train_dist = get_class_distribution_loaders(train_loader, dataset)
val_dist = get_class_distribution_loaders(val_loader, dataset)
plot_class_dir_loaders(train_dist, val_dist)
###

#########  WeightedRandomSampler
# using this is relevant only for the train-loader. should be performed on train-dataset only

target_list = torch.tensor(dataset.targets)

# Get the class counts and calculate the weights/class by taking its reciprocal.
class_count = [i for i in get_class_distribution(dataset).values()] # Both are identical
class_count = list(get_class_distribution(dataset).values()) # Both are identical

class_weights = 1./torch.tensor(class_count, dtype=torch.float)

# Assign the weight of each class to all the samples
class_weights_all = class_weights[target_list] # sum(class_weights_all) = N_classes

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=8, sampler=weighted_sampler)

sum_0, sum_1 = 0, 0
for idx, (data,label) in enumerate(train_loader):
    sum_0 += sum(label == 0)
    sum_1 += sum(label == 1)
    print(f"batch index: {idx}, class 0: {sum(label == 0)}, class 1: {sum(label == 1)}")
print(f"sum class 0: {sum_0}, sum class 1: {sum_1}")


it = iter(train_loader)
data, labels = it.__next__()
### end of "like example"



# Or (for specifically for the train_set) - it can't be used with a splitted dataset!
train_target_list = []
train_class_count = [0,0]
for idx, (data, label) in enumerate(train_dataset):
    print(f"idx: {idx}, label: {label}")
    train_target_list.append(label)
    train_class_count[label] += 1

train_class_weights = 1./torch.tensor(train_class_count, dtype=torch.float)
train_class_weights_all = train_class_weights[train_target_list] # sum(class_weights_all) = N_classes

weighted_sampler = WeightedRandomSampler(
    weights=train_class_weights_all,
    num_samples=len(train_class_weights_all),
    replacement=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10,sampler=, num_workers=0)

for idx, (data,label) in enumerate(train_loader):
    print(f"label: {label}")
    print(f"class 0: {sum(label == 0)}, class 1: {sum(label == 1)}")


# Basically the same for SubsetRandomSampler
# https://stackoverflow.com/questions/67250023/related-to-subsetrandomsampler


# Or in more length (plus calculate data mean and std (of images)
# https://blogs.oracle.com/ai-and-datascience/post/transfer-learning-in-pytorch-part-1-how-to-use-dataloaders-and-build-a-fully-connected-class