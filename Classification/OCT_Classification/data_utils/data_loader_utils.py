import os
import torch

from torchvision import transforms
from torchvision.utils import make_grid, save_image
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

input_size = 192

output_dir_imgs = "/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/dataset_images/"
output_dir_grid = "/home/nim/venv/DL-code/Classification/OCT_Classification/data_utils/grid_images"

###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
#

def get_dataset_metadata(dataset):
    # get dataset's metadata
    print(f'train_dataset.root: {dataset.root}')
    print(f'len(train_dataset.imgs): {len(dataset.imgs)}')
    print(f'train_dataset.__len__(): {dataset.__len__()}')
    print(f'train_dataset.classes: {dataset.classes}')
    print(f'train_dataset.class_to_idx: {dataset.class_to_idx}')
    print(f'train_dataset.extensions: {dataset.extensions}')
    print(f'train_dataset.__getitem__(0)[0].shape: {dataset.__getitem__(0)[0].shape}')
    print(f'train_dataset.__getitem__(0)[1]: {dataset.__getitem__(0)[1]}')
    print(f'train_dataset.imgs: {dataset.imgs}') # samples and images are both lists [(path, target), (path, target)...]
    print(f'train_dataset.samples: {dataset.samples}')
    print(f'train_dataset.targets: {dataset.targets}')
    print(f'train_dataset.transforms: {dataset.transforms}')

def denormalize_img_tensor_plot(img_tensor, plot=True):
    # denormalize img_tensor and convert to a PIL image, and then plot the PIL image

    # Convert the normalized image tensor to a NumPy array
    img_denormalized = img_tensor.cpu().numpy().transpose((1, 2, 0))

    # Denormalize the image tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_denormalized = (std * img_denormalized) + mean
    img_denormalized = np.clip(img_denormalized, 0, 1)

    # Convert the denormalized array to a PIL Image
    img_denormalized = (img_denormalized * 255).astype(np.uint8)  # Convert to uint8
    img_pil = Image.fromarray(img_denormalized)

    if plot:
        plt.imshow(img_pil)
    return img_denormalized, img_pil

# OR
def denormalize_img_tensor_plot2(img_tensor, plot=True):
    # denormalize img_tensor and convert to a PIL image, and then plot the PIL image
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    ])
    img_denormalized = denormalize(img_tensor).clamp(0, 1)

    # Convert the denormalized tensor to a PIL Image
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_denormalized)

    if plot:
        plt.imshow(img_pil)

    return img_denormalized, img_pil

def denormalize(tensor):
    # denormalize image tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)  # Convert to uint8
    return img

def save_dataset_images(dataset, output_dir):
    # save all the images in the dataset
    os.makedirs(output_dir, exist_ok=True)
    img_num = 0
    for img, label in dataset:
        img_denormalized, img_pil = plot_img_tensor_denormalize2(img, plot=True)
        plt.imshow(img_pil)
        plt.savefig(output_dir + str(img_num) + '.png')
        img_num += 1
        print('img: ', img_num)


def plot_dataloder_as_grid(dataloader, output_dir_grid):
    # Plot images in the dataloader in a grid (the grid looks somewhat "white")
    os.makedirs(output_dir_grid, exist_ok=True)

    # Iterate over the train_loader
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Create a grid of images
        grid = make_grid(images, nrow=3, padding=5, normalize=True)

        # Convert the normalized grid to a numpy array
        grid_np = denormalize(grid)

        # Add true class names as text annotations
        for i, label in enumerate(labels):
            row = i // 3  # Assuming nrow=3 in make_grid
            col = i % 3
            true_class_name = train_dataset.classes[label.item()]
            plt.text(
                col * (input_size + 5) + 5,
                row * (input_size + 5) + 5,
                f"lbl: {true_class_name}",
                color='white',
                backgroundcolor='black',
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left'
            )

        # Display the grid with labels
        plt.imshow(grid_np)
        plt.axis('off')

        # Save the grid as an image
        image_path = os.path.join(output_dir_grid, f"grid_batch_{batch_idx}.png")
        plt.savefig(image_path)
        plt.close()

    print(f"Grid images with true labels saved in {output_dir_grid}")
####

transform = transforms.Compose([
    transforms.Resize((input_size,input_size)), # (h,w)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(train_data_path, transform=transform)
val_dataset = ImageFolder(val_data_path, transform=transform)
get_dataset_metadata(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6,
                                          shuffle=False, num_workers=2)


##### dataset - plot image
img_i = 0
img_path = train_dataset.imgs[img_i][0] # str
label = train_dataset.__getitem__(img_i)[1] # int
img_tensor = train_dataset.__getitem__(img_i)[0] #  type(img_tensor) - torch.Tensor, img_tensor.dtype - torch.float32

img_denormalized, img_pil = denormalize_img_tensor_plot(img_tensor, plot=True)

# save_dataset_images(train_dataset, output_dir_imgs) # Save the whole train_dataset

##### dataloader - plot image
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs, labels = data[0].to(device), data[1].to(device)  # inputs.dtype and labels.dtype - torch.int64
    print(f'inputs.shape: {inputs.shape}, labels: {labels.shape}')
    if i == 3:
        break
denormalize_img_tensor_plot(inputs[0]) # plot the first image of the current batch (also normalize the image)

# plot images from dataloader on a grid, and save grid images
plot_dataloder_as_grid(train_loader, output_dir_grid)

