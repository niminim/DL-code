import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_file, base_data_folder, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image file paths, phase, label, width, and height information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, index_col=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        phase = self.data_frame.iloc[idx, 1]
        label = self.data_frame.iloc[idx, 2]
        width = int(self.data_frame.iloc[idx, 3])
        height = int(self.data_frame.iloc[idx, 4])
        filepath = os.path.join(base_data_folder,phase, label, img_name)

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

base_data_folder = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_0_01' # data path
train_csv_file_path = '/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/csv_splits/train_split_0_01.csv' # csv of same data
val_csv_file_path = '/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/csv_splits/val_split_0_01.csv' # csv of same data
save_imgs_dir = '/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/custom_dataset_images' # dir  to save samples from the dataset


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust as needed
    # transforms.RandomRotation((20)),  # Adjust as needed
    transforms.ToTensor(),
])

# Create an instance of your custom dataset
train_dataset = CustomDatasetFromCSV(csv_file=train_csv_file_path, base_data_folder=base_data_folder, transform=transform)
val_dataset = CustomDatasetFromCSV(csv_file=val_csv_file_path, base_data_folder=base_data_folder, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24,
                                          shuffle=False, num_workers=2)

# Print the instance metadata and plot the image
plot_sample_by_index(train_dataset, index=10)

# Save images from the dataset
# save_dataset_imgs(dataset=train_dataset, save_path=save_imgs_dir, num_imgs=50)

it = iter(train_loader)
it.__next__()[0].shape # torch.Size([24, 1, 224, 224])




# https://github.com/utkuozbulak/pytorch-custom-dataset-examples


