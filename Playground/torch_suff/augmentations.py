from pathlib import Path
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

plt.rcParams["savefig.bbox"] = 'tight'

import torchvision
from torchvision.transforms import v2
from torchvision.io import read_image

torch.manual_seed(1)

img = read_image('/home/nim/Downloads/image001.png')
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")



def plot_aug_img(img, aug):
    # perform transform and plot the transformed image
    
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




######  Inspecting the RGBA image
img = read_image('/home/nim/Downloads/image001.png')
alpha_channel = img[3, :, :]  # Extract the 4th channel (Alpha channel)
print(alpha_channel.shape)  # Should be [H, W], the height and width of the image
# Display the Alpha channel
plt.imshow(alpha_channel, cmap='gray')  # Alpha channel is a grayscale image
plt.title('Alpha (Transparency) Channel')


# Remove the alpha channel (take only the first 3 channels: RGB)
img_rgb = img[:3, :, :]

# Convert the tensor back to a PIL image (normalize to [0, 1] range)
# torchvision.io.read_image loads in [0, 255], but ToPILImage expects [0, 1] range
to_pil = torchvision.transforms.ToPILImage()

img_rgb_pil = to_pil(img_rgb / 255.0)  # Normalize the tensor to [0, 1] for PIL conversion

# Save the image in RGB format
img_rgb_pil.save('/home/nim/Downloads/converted_rgb_image.png')
######