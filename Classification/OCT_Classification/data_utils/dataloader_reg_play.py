import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from Classification.OCT_Classification.data_utils.data_loader_utils import create_dataloader, get_class_dist_from_dataloader, get_class_dist_from_dataset
# it runs the whole module if there's no if == main

matplotlib.use('Qt5Agg')

### The following code simply uses ImageDaraFolder to create the train and val datasets
# Then it creates dataloaders in different Class-Distributions

# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

#####
data_path = "/home/nim/Downloads/cats_and_dogs/train"

data_path = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_0_01/train'

###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


transform = transforms.Compose([
    transforms.Resize((224,224)), # (h,w)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset = ImageFolder(data_path, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           shuffle=True, num_workers=2)

num_classes = len(dataset.classes)

class_indices, class_dist, idx2class = get_class_dist_from_dataset(dataset, num_classes)
class_counts = get_class_dist_from_dataloader(data_loader, num_classes)

# plot the dataset distribution
plt.figure(figsize=(15,8))
sns.barplot(data = pd.DataFrame.from_dict([class_dist]).melt(),
            x = "variable", y="value", hue="variable").set_title('Class Distribution')
####


### Create dataloaders using random_split (for get_class_distribution_loaders implementation must equal 1)
# Beware of having one transform for both datasets!
# Keep in mind the train-set is addressed as the whole dataset here.
# Keep in mind the train-set is addressed as the whole dataset here.
# random_split transform the "dataset" into "dataset.Subset" which has no ".transform" --> random_split is not very useful

n_imgs_train = int(len(dataset.samples) * 0.8)
n_imgs_val = int(len(dataset.samples) - n_imgs_train)
train_dataset, val_dataset = random_split(dataset, (n_imgs_train, n_imgs_val))
train_loader, val_loader = create_dataloader(train_dataset, val_dataset, bs_train=1, bs_val=1) # bs must equal for to comply with func that calculates claas-dist

class_counts_train = get_class_dist_from_dataloader(train_loader, num_classes=2)
class_counts_val = get_class_dist_from_dataloader(val_loader, num_classes=2)

def plot_class_dir_loaders(train_dist, val_dist):
    # plots the class distribution in the train and val sets
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,7))
    sns.barplot(data = pd.DataFrame.from_dict([train_dist]).melt(),
                x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Train Set')
    sns.barplot(data = pd.DataFrame.from_dict([val_dist]).melt(),
                x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Val Set')

# Show Class-Distribution in the new random train-val splits
plot_class_dir_loaders(class_counts_train, class_counts_val)
#######


#########  SubsetRandomSampler
# Here we split the dataset to train and val (by indices)
dataset_size = len(dataset)
dataset_indices = list(range(dataset_size))
np.random.shuffle(dataset_indices)
val_split_index = int(np.floor(0.2 * dataset_size))
train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=train_sampler)
val_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=val_sampler)
# val_loader.dataset.transform = transform_val # if necessary

class_counts_train = get_class_dist_from_dataloader(train_loader, num_classes=2)
class_counts_val = get_class_dist_from_dataloader(val_loader, num_classes=2)
plot_class_dir_loaders(class_counts_train, class_counts_val)
###

#########  WeightedRandomSampler (this is how to do it)
# using this is relevant only for the train-loader. should be performed on train-dataset only
# train-dataset should be created in advance (without indexing) if we want to use WeightedRandomSampler

target_list = torch.tensor(dataset.targets)

# Get the class counts and calculate the weights/class by taking its reciprocal.
class_indices, class_dist, idx2class = get_class_dist_from_dataset(dataset, num_classes)
class_counts = get_class_dist_from_dataloader(data_loader, num_classes)

values_list = list(class_counts.values())
class_weights = 1./torch.tensor(values_list, dtype=torch.float)

# Assign the weight of each class to all the samples
class_weights_all = class_weights[target_list] # sum(class_weights_all) = N_classes

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

train_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=32,sampler=weighted_sampler)

class_counts_by_batch = {i: 0 for i in range(num_classes)}
for idx, (data,label) in enumerate(train_loader):
    for k,v in class_counts_by_batch.items():
        class_counts_by_batch[k] += sum(label == k).item()
    print(f"batch index: {idx}, class 0: {class_counts_by_batch[0]}, class 1: {class_counts_by_batch[1]}, class 2: {class_counts_by_batch[2]}, class 3: {class_counts_by_batch[3]}")

### Only for cats and dogs
sum_0, sum_1 = 0, 0
for idx, (data,label) in enumerate(train_loader):
    sum_0 += sum(label == 0)
    sum_1 += sum(label == 1)
    print(f"batch index: {idx}, class 0: {sum(label == 0)}, class 1: {sum(label == 1)}")
print(f"sum class 0: {sum_0}, sum class 1: {sum_1}")
####

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

class_indices, class_dist, idx2class = get_class_dist_from_dataset(train_dataset, num_classes=2)

train_class_weights = 1./torch.tensor(train_class_count, dtype=torch.float)
train_class_weights_all = train_class_weights[train_target_list] # sum(class_weights_all) = N_classes

weighted_sampler = WeightedRandomSampler(
    weights=train_class_weights_all,
    num_samples=len(train_class_weights_all),
    replacement=True
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10,sampler=weighted_sampler, num_workers=0)

for idx, (data,label) in enumerate(train_loader):
    print(f"label: {label}")
    print(f"class 0: {sum(label == 0)}, class 1: {sum(label == 1)}")



# https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
# https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
# https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
# # https://blogs.oracle.com/ai-and-datascience/post/transfer-learning-in-pytorch-part-1-how-to-use-dataloaders-and-build-a-fully-connected-class (plus calculate data mean and std (of images)