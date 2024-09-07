from pathlib import Path
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')  # or 'Qt5Agg' depending on your system
plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.transforms import v2
from torchvision.io import read_image

torch.manual_seed(1)

img = read_image('/home/nim/Downloads/קומה 13 דירה 55.jpeg')
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")


def plot_aug_img(img, aug):
    out = aug(img)

    # Convert the image to a format that plt.imshow can understand
    # The image needs to be permuted from (C, H, W) to (H, W, C) format
    out_permuted = out.permute(1, 2, 0)

    # Display the cropped image using imshow
    plt.imshow(out_permuted)
    plt.axis('off')  # Hide axis
    plt.show()

transform_crop = v2.RandomCrop(size=(224, 224))

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
])

plot_aug_img(img, aug=transform_crop)