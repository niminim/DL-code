import os.path

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

        return {'img': img, 'img_name': img_name, 'filepath': filepath, 'phase': phase, 'label': label, 'width': width, 'height': height}

def plot_sample_by_index(index):
    sample = custom_dataset.__getitem__(index)
    # Print the details of the sample
    print(
        f"Image shape: {sample['img'].shape}, Image name: {sample['img_name']}, Phase: {sample['phase']}, Label: {sample['label']}, "
        f"Width: {sample['width']}, Height: {sample['height']}")
    # plot the image
    plt.imshow(transforms.ToPILImage()(sample['img']), cmap='gray') # default for matplotlib is color, for a grayscale need to assign 'gray' in the cmap
    plt.title(f"Phase: {sample['phase']}, Label: {sample['label']}")
    plt.show()

    # # Plot the image using Image.show()
    # pil_img = transforms.ToPILImage()(sample['img'])
    # pil_img.show()

base_data_folder = '/home/nim/Downloads/OCT_and_X-ray/OCT2017/train_split_0_035' # data path
csv_file_path = '/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/csv_splits/train_split.csv' # csv of same data

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust as needed
    # transforms.RandomRotation((20)),  # Adjust as needed
    transforms.ToTensor(),
])

# Create an instance of your custom dataset
custom_dataset = CustomDatasetFromCSV(csv_file=csv_file_path, base_data_folder=base_data_folder, transform=transform)
# Print the instance metadata and plot the image
plot_sample_by_index(index=10)


path = '/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/custom_dataset_images'

for i in range(50):
    sample = custom_dataset.__getitem__(i)
    plt.imshow(transforms.ToPILImage()(sample['img']), cmap='gray')
    plt.title(f"Phase: {sample['phase']}, Label: {sample['label']}")
    plt.savefig(os.path.join(path,sample['img_name']))


# https://github.com/utkuozbulak/pytorch-custom-dataset-examples


