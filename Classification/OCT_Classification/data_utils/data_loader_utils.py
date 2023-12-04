import torch
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import matplotlib
matplotlib.use('Qt5Agg')

#####
train_data_path = "/home/nim/Downloads/cats_and_dogs/train"
val_data_path = "/home/nim/Downloads/cats_and_dogs/val"

input_size = 192
model_name = 'efficientnet' # mobilenet, efficientnet

def get_dataset_metadata(dataset):
    print(f'train_dataset.root: {dataset.root}')
    print(f'len(train_dataset.imgs): {len(dataset.imgs)}')
    print(f'train_dataset.__len__(): {dataset.__len__()}')
    print(f'train_dataset.classes: {dataset.classes}')
    print(f'train_dataset.extensions: {dataset.extensions}')
    print(f'train_dataset.__getitem__(0)[0].shape: {dataset.__getitem__(0)[0].shape}')
    print(f'train_dataset.__getitem__(0)[1]: {dataset.__getitem__(0)[1]}')
    print(f'train_dataset.imgs: {dataset.imgs}') # samples and images are both lists [(path, target), (path, target)...]
    print(f'train_dataset.samples: {dataset.samples}')
    print(f'train_dataset.targets: {dataset.targets}')
    print(f'train_dataset.transforms: {dataset.transforms}')


def imshow_np(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose([
    transforms.Resize((input_size,input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = ImageFolder(train_data_path, transform=transform)
val_dataset = ImageFolder(val_data_path, transform=transform)
get_dataset_metadata(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                          shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=6,
                                          shuffle=False, num_workers=2)

######
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

img_i = 0
img_path = train_dataset.imgs[img_i][0] # str
label = train_dataset.__getitem__(img_i)[1] # int
img_tensor = train_dataset.__getitem__(img_i)[0] #  type(img_tensor) - torch.Tensor, img_tensor.dtype - torch.float32

import torchvision.transforms.functional as F
img_pil = F.to_pil_image(img_tensor)

def print_img_tensor_denormalize(img_tensor):
    # Print img_tensor as an image
    import torchvision.transforms.functional as F

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

    # Display the image
    plt.imshow(img_pil)

# OR
def print_img_tensor_denormalize2(img_tensor):
    # Denormalize the image tensor
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    ])
    img_denormalized = denormalize(img_tensor).clamp(0, 1)

    # Convert the denormalized tensor to a PIL Image
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_denormalized)

    # Display the image
    plt.imshow(img_pil)



for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    inputs, labels = data[0].to(device), data[1].to(device)  # inputs.dtype and labels.dtype - torch.int64
    print(f'inputs.shape: {inputs.shape}, labels: {labels.shape}')
    if i == 3:
        break

### PIL Image
img = Image.open(img_path) # type(img) = PIL Image
W, H = img.size
print(f'size - H: {H}, W: {W}')
img.show() # Show with PIL
plt.imshow(img) # show with matplotlib (viewer)


### opencv Image
img_cv2 = cv2.imread(img_path) # type(img) = np.ndarray, img.dtype = uint8
H, W, ch = img_cv2.shape
print(f'opencv image: size - H: {H}, W: {W}, Ch: {ch}')
plt.imshow(img_cv2)
plt.show()

# Numpy data
img_data = np.asarray(img)
H, W, Ch = img_data.shape
print(f'size - H: {H}, W: {W}')
print(f'dtype: {img_data.dtype}')
plt.imshow(img_data)
plt.show()


npimg = img.numpy()


img = img / 2 + 0.5  # unnormalize
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))

