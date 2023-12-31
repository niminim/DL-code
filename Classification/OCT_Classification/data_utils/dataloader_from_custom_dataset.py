import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

from Classification.OCT_Classification.data_utils.dataset_transforms import get_one_ch_transform
from Classification.OCT_Classification.data_utils.data_loader_utils import create_dataloader, get_class_dist_from_dataloader, get_class_dist_from_dataset
# it runs the whole module if there's no if == main

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_file, base_data_folder, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image file paths, phase, label, width, and height information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, index_col=None)
        self.base_data_folder = base_data_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        phase = self.data_frame.iloc[idx, 1]
        label = self.data_frame.iloc[idx, 2]
        width = int(self.data_frame.iloc[idx, 3])
        height = int(self.data_frame.iloc[idx, 4])
        filepath = os.path.join(base_data_folder, phase, label, img_name)

        # Open the image file
        img = Image.open(filepath)

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        img_data = {'img_name': img_name, 'filepath': filepath, 'phase': phase, 'width': width, 'height': height}
        return img, label, img_data

def plot_sample_by_index(dataset, index):
    # The function gets a dataset and an index
    # The function prints the metadata of image of index "idx" from the input dataset, and plots the image
    assert index < len(dataset)

    img, label, img_data = dataset.__getitem__(index)
    # Print the details of the sample
    print(
        f"Image shape: {img.shape}, Image name: {img_data['img_name']}, Phase: {img_data['phase']}, Label: {label}, "
        f"Width: {img_data['width']}, Height: {img_data['height']}")
    # plot the image
    plt.imshow(transforms.ToPILImage()(img), cmap='gray') # default for matplotlib is color, for a grayscale need to assign 'gray' in the cmap
    plt.title(f"Phase: {img_data['phase']}, Label: {label}")
    plt.show()

    # # Plot the image using Image.show()
    # pil_img = transforms.ToPILImage()(sample['img'])
    # pil_img.show()


def save_dataset_imgs(dataset, save_dir, num_imgs):
    # The function gets a dataset and saves num_imgs images in save_dir

    for i in range(num_imgs):
        img, label, img_data = dataset.__getitem__(i)
        plt.imshow(transforms.ToPILImage()(img), cmap='gray')
        plt.title(f"Phase: {img_data['phase']}, Label: {label}")
        plt.savefig(os.path.join(save_dir, img_data['img_name']))

split = '0_01'
# split = '0_035'

main_data_utils_dir = '/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils'
base_data_folder = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_' + split # data path
train_csv_file_path = os.path.join(main_data_utils_dir, 'csv_splits/train_split_' + split + '.csv')# csv of same data
val_csv_file_path = os.path.join(main_data_utils_dir, 'csv_splits/val_split_' + split + '.csv')# # csv of same data
save_imgs_dir = os.path.join(main_data_utils_dir,'custom_dataset_images') # dir  to save samples from the dataset


transform = get_one_ch_transform(input_size=224)

# Create an instance of your custom dataset
train_dataset = CustomDatasetFromCSV(csv_file=train_csv_file_path, base_data_folder=base_data_folder, transform=transform)
val_dataset = CustomDatasetFromCSV(csv_file=val_csv_file_path, base_data_folder=base_data_folder, transform=transform)

train_loader, val_loader = create_dataloader(train_dataset, val_dataset, bs_train=24, bs_val=24)

def get_class_dist_from_dataloader_custom_csv(data_loader, csv_file_path):
    # the function calculates the class distribution of a dataloader
    df_train = pd.read_csv(csv_file_path, index_col=None)
    labels = list(df_train['label'].unique())
    class_counts = {i: 0 for i in labels}

    for imgs, labels, img_data in data_loader:
        for label in labels:
            class_counts[label] += 1
    print(f'class_counts from dataloder: {class_counts}')

    return class_counts

class_counts = get_class_dist_from_dataloader_custom_csv(train_loader, train_csv_file_path)

# Print the instance metadata and plot the image
plot_sample_by_index(train_dataset, index=10)

# Save images from the dataset
# save_dataset_imgs(dataset=train_dataset, save_path=save_imgs_dir, num_imgs=50)

# it = iter(train_loader)
# it.__next__()[0].shape # torch.Size([24, 1, 224, 224])


# https://github.com/utkuozbulak/pytorch-custom-dataset-examples


